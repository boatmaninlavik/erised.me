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

## Deployment to erised.me (GitHub Pages)

Your files are already pushed to GitHub at: https://github.com/boatmaninlavik/erised.me

### Step 1: Enable GitHub Pages

1. Go to your repository: https://github.com/boatmaninlavik/erised.me
2. Click **Settings** (top right of the repo)
3. Scroll down to **Pages** in the left sidebar
4. Under **Source**, select **Deploy from a branch**
5. Choose **main** branch
6. Select **/ (root)** folder
7. Click **Save**

Your site will be available at: `https://boatmaninlavik.github.io/erised.me/`

### Step 2: Connect Custom Domain (erised.me)

1. In the same **Pages** settings section
2. Under **Custom domain**, enter: `erised.me`
3. Check **Enforce HTTPS** (once DNS is configured)

### Step 3: Configure DNS

In your domain registrar (where you bought erised.me), add these DNS records:

**Option A: A Records (Recommended)**
```
Type: A
Name: @
Value: 185.199.108.153
TTL: 3600

Type: A
Name: @
Value: 185.199.109.153
TTL: 3600

Type: A
Name: @
Value: 185.199.110.153
TTL: 3600

Type: A
Name: @
Value: 185.199.111.153
TTL: 3600
```

**Option B: CNAME Record**
```
Type: CNAME
Name: @ (or www)
Value: boatmaninlavik.github.io
TTL: 3600
```

**Note:** DNS changes can take up to 48 hours to propagate, but often work within minutes.

### Verify

After DNS propagates, visit https://erised.me to see your site live!

