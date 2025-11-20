# ERISED AI Website

Minimal, Scandinavian-style website for erised.me

## Setup

1. Add your logo image file as `logo.png` in the root directory

## Running Locally

### Option 1: Simple (Double-click)
- Double-click `index.html` to open it in your default browser

### Option 2: Local Server (Recommended)
Open Terminal in this directory and run:

```bash
# Using Python 3 (most common)
python3 -m http.server 8000

# Or using Python 2
python -m SimpleHTTPServer 8000
```

Then open your browser and go to: `http://localhost:8000`

To stop the server, press `Ctrl+C` in the terminal.

## Files

- `index.html` - Main page structure
- `styles.css` - Styling
- `logo.png` - Company logo (you need to add this file)

## Deployment to erised.me (Vercel or Netlify)

Your files are already pushed to GitHub at: https://github.com/boatmaninlavik/erised.me

### Option A: Deploy with Vercel (Recommended)

**Step 1: Import Project**
1. Go to: https://vercel.com
2. Sign in with GitHub
3. Click **Add New...** → **Project**
4. Import the repository: `boatmaninlavik/erised.me`
5. Vercel will auto-detect it as a static site
6. Click **Deploy**

Your site will be live at: `https://erised-me.vercel.app` (or similar URL)

**Step 2: Add Custom Domain**
1. Go to your project dashboard in Vercel
2. Click **Settings** → **Domains**
3. Enter: `erised.me`
4. Click **Add**
5. Vercel will show you the DNS records to add

**Step 3: Configure DNS at Porkbun**
1. Log into Porkbun: https://porkbun.com
2. Go to your domain: erised.me → DNS
3. Add the CNAME record that Vercel provides:
   ```
   Type: CNAME
   Name: @ (or blank for root)
   Answer: cname.vercel-dns.com (or what Vercel shows)
   TTL: 3600
   ```
   OR if Vercel provides A records, add those instead.

**Step 4: Wait for DNS**
- DNS propagation usually takes 5-30 minutes
- Vercel will automatically issue SSL certificate once DNS is verified

---

### Option B: Deploy with Netlify

**Step 1: Import Project**
1. Go to: https://app.netlify.com
2. Sign in with GitHub
3. Click **Add new site** → **Import an existing project**
4. Choose **Deploy with GitHub**
5. Select repository: `boatmaninlavik/erised.me`
6. Click **Deploy site**

Your site will be live at: `https://random-name.netlify.app`

**Step 2: Add Custom Domain**
1. Go to **Site configuration** → **Domain management**
2. Click **Add custom domain**
3. Enter: `erised.me`
4. Click **Verify**

**Step 3: Configure DNS at Porkbun**
1. Log into Porkbun: https://porkbun.com
2. Go to your domain: erised.me → DNS
3. Add the DNS record Netlify provides (usually a CNAME):
   ```
   Type: CNAME
   Name: @
   Answer: (what Netlify shows - usually ends with netlify.app)
   TTL: 3600
   ```

**Step 4: Wait for DNS**
- Netlify will automatically provision SSL once DNS is verified

---

### Verify Deployment

After DNS propagates (5-30 minutes), visit https://erised.me to see your site live!

