#!/usr/bin/env bash
# One-time setup plus deploy of the service to Cloud Run.
#
#   export PROJECT_ID=my-project REGION=europe-west3
#   export CUSTOMER_MODELS=openai/<fast-model>,gemini/<fast-model>
#   export COACH_MODELS=anthropic/<strong-model>,openai/<strong-model>
#   export JUDGE_MODELS=openai/<strong-model>
#   ./deploy/deploy.sh
#
# Re-running is safe: every create step skips what already exists.
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-europe-west3}"
REPO="${REPO:-coach}"
SERVICE="conversation-coach-agent"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:${TAG}"
RUNTIME_SA="coach-runtime@${PROJECT_ID}.iam.gserviceaccount.com"
CUSTOMER_MODELS="${CUSTOMER_MODELS:?set CUSTOMER_MODELS}"
COACH_MODELS="${COACH_MODELS:?set COACH_MODELS}"
JUDGE_MODELS="${JUDGE_MODELS:?set JUDGE_MODELS}"
ALERT_EMAIL="${ALERT_EMAIL:-}"

gcloud config set project "$PROJECT_ID" >/dev/null

echo "==> Enabling APIs"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com monitoring.googleapis.com logging.googleapis.com

echo "==> Artifact Registry repository"
gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 ||
  gcloud artifacts repositories create "$REPO" --repository-format docker --location "$REGION"

echo "==> Runtime service account (least privilege: read its own secrets, write logs and metrics)"
gcloud iam service-accounts describe "$RUNTIME_SA" >/dev/null 2>&1 ||
  gcloud iam service-accounts create coach-runtime --display-name "Conversation coach runtime"

for secret in openai-api-key anthropic-api-key; do
  if ! gcloud secrets describe "$secret" >/dev/null 2>&1; then
    echo "==> Secret $secret does not exist; create it with:"
    echo "    printf '%s' \"\$KEY\" | gcloud secrets create $secret --data-file=-"
    exit 1
  fi
  gcloud secrets add-iam-policy-binding "$secret" --member "serviceAccount:${RUNTIME_SA}" \
    --role roles/secretmanager.secretAccessor >/dev/null
done

echo "==> Building $IMAGE"
gcloud builds submit --config deploy/cloudbuild.yaml \
  --substitutions "_REGION=${REGION},_REPO=${REPO},_TAG=${TAG}" .

echo "==> Deploying"
rendered="$(mktemp)"
sed -e "s|IMAGE|${IMAGE}|" -e "s|PROJECT_ID|${PROJECT_ID}|" \
  -e "s|CUSTOMER_MODELS_VALUE|${CUSTOMER_MODELS}|" -e "s|COACH_MODELS_VALUE|${COACH_MODELS}|" \
  -e "s|JUDGE_MODELS_VALUE|${JUDGE_MODELS}|" deploy/cloudrun-service.yaml >"$rendered"
gcloud run services replace "$rendered" --region "$REGION"
rm -f "$rendered"

if [[ "${ALLOW_UNAUTHENTICATED:-false}" == "true" ]]; then
  gcloud run services add-iam-policy-binding "$SERVICE" --region "$REGION" \
    --member allUsers --role roles/run.invoker
fi

echo "==> Log-based metrics from the service's JSON logs"
gcloud logging metrics describe coach_llm_failures >/dev/null 2>&1 ||
  gcloud logging metrics create coach_llm_failures \
    --description "LLM calls that failed (before retry or fallback)" \
    --log-filter "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE}\" AND jsonPayload.message=\"llm call failed\""
gcloud logging metrics describe coach_feedback_failures >/dev/null 2>&1 ||
  gcloud logging metrics create coach_feedback_failures \
    --description "Feedback requests that returned 503" \
    --log-filter "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE}\" AND jsonPayload.message=\"feedback failed\""

if [[ -n "$ALERT_EMAIL" ]]; then
  echo "==> Alert policies (notifications to $ALERT_EMAIL)"
  channel="$(gcloud beta monitoring channels list --filter "labels.email_address=${ALERT_EMAIL}" \
    --format 'value(name)' | head -n1)"
  if [[ -z "$channel" ]]; then
    channel="$(gcloud beta monitoring channels create --display-name "Coach on-call" --type email \
      --channel-labels "email_address=${ALERT_EMAIL}" --format 'value(name)')"
  fi
  for policy in deploy/monitoring/*.json; do
    name="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['displayName'])" "$policy")"
    if ! gcloud alpha monitoring policies list --filter "displayName=\"${name}\"" --format 'value(name)' | grep -q .; then
      sed -e "s|SERVICE_NAME|${SERVICE}|g" "$policy" >"${policy}.rendered"
      gcloud alpha monitoring policies create --policy-from-file "${policy}.rendered" --notification-channels "$channel"
      rm -f "${policy}.rendered"
    fi
  done
fi

url="$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')"
echo "==> Deployed: $url"
echo "    Smoke test:  curl -s $url/healthz"
echo "    Load test:   python -m loadtest.run_load --url $url --conversations 50 --concurrency 10"
