# ERISED AI Backend

Express.js backend API for ERISED AI, deployed on Google Cloud Run.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Copy environment variables:
```bash
cp .env.example .env
```

3. Fill in your `.env` file with your actual values

4. Run locally:
```bash
npm run dev
```

Server will run on `http://localhost:3000`

## Deploy to Google Cloud Run

### Prerequisites
- Google Cloud account with billing enabled
- Google Cloud SDK (gcloud) installed
- Docker installed

### Deployment Steps

#### Option 1: Using Cloud Build (Recommended)

1. **Set up Google Cloud project:**
```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

2. **Deploy using Cloud Build:**
```bash
gcloud builds submit --config cloudbuild.yaml
```

#### Option 2: Manual Deployment

1. **Build Docker image:**
```bash
docker build -t gcr.io/YOUR_PROJECT_ID/erised-backend .
```

2. **Push to Container Registry:**
```bash
docker push gcr.io/YOUR_PROJECT_ID/erised-backend
```

3. **Deploy to Cloud Run:**
```bash
gcloud run deploy erised-backend \
  --image gcr.io/YOUR_PROJECT_ID/erised-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Set Environment Variables in Cloud Run

After deployment, set your environment variables:

```bash
gcloud run services update erised-backend \
  --update-env-vars "BETTER_AUTH_SECRET=your-secret,PORT=3000" \
  --region us-central1
```

Or use the Cloud Console:
1. Go to Cloud Run → erised-backend → Edit & Deploy New Revision
2. Add environment variables in the "Variables & Secrets" tab
3. Deploy

### Custom Domain Setup

1. In Cloud Run dashboard, click on your service
2. Go to "Custom Domains" tab
3. Add `api.erised.me` (or your preferred subdomain)
4. Follow DNS instructions to add CNAME record in Porkbun

## API Endpoints

- `GET /health` - Health check
- `GET /api` - API information

## Notes

- Cloud Run automatically scales to zero when not in use
- Pay only for actual usage
- Supports up to 60 minutes request timeout
- Automatic HTTPS/SSL certificates

