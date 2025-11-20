# Deploy erised.me to Google Cloud Run

Follow these steps to deploy the website to Google Cloud Run using Express.js.

## Prerequisites

1. **Google Cloud Account**
   - Sign up at: https://cloud.google.com (Free $300 credit)
   - Create a new project

2. **Install Google Cloud SDK**
   ```bash
   # Mac
   brew install google-cloud-sdk
   
   # Or download from: https://cloud.google.com/sdk/docs/install
   ```

3. **Login and setup**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

## Enable Required Services

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## Deploy to Cloud Run

### Option 1: Using Cloud Build (Automated)

From the root directory of the project:

```bash
gcloud builds submit --config cloudbuild.yaml
```

This will:
- Build the Docker image
- Push to Container Registry
- Deploy to Cloud Run automatically

### Option 2: Manual Deployment

1. **Build Docker image:**
   ```bash
   docker build -t gcr.io/YOUR_PROJECT_ID/erised-website .
   ```

2. **Push to Container Registry:**
   ```bash
   docker push gcr.io/YOUR_PROJECT_ID/erised-website
   ```

3. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy erised-website \
     --image gcr.io/YOUR_PROJECT_ID/erised-website \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --port 3000
   ```

## Setup Custom Domain (erised.me)

1. **In Cloud Run Console:**
   - Go to: https://console.cloud.google.com/run
   - Click on your service: `erised-website`
   - Go to "Custom Domains" tab
   - Click "Add Custom Domain"
   - Enter: `erised.me`

2. **Configure DNS in Porkbun:**
   - Cloud Run will show you the DNS records to add
   - Usually it's a CNAME or A records
   - Add them in Porkbun DNS settings

3. **Wait for DNS propagation** (5-30 minutes)

4. **Verify:** Visit https://erised.me

## Team Access

- **Add team members in Google Cloud Console:**
  - Go to IAM & Admin → IAM
  - Click "Grant Access"
  - Add email addresses
  - Grant "Cloud Run Admin" or "Cloud Run Developer" role
  - **FREE** - No paid features needed!

## Notes

- Cloud Run auto-scales to zero when not in use
- Pay only for actual usage
- Free tier includes 2 million requests per month
- Automatic HTTPS/SSL certificates
- Multiple team members can access for free

