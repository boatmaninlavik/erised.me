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

## Deployment to erised.me

To make your website live at https://erised.me, you need to upload these files to your web hosting:

### Files to Upload:
- `index.html` (must be in the root/public_html directory)
- `styles.css` (same directory as index.html)

### Common Hosting Methods:

**1. FTP/SFTP Upload:**
- Connect to your hosting via FTP client (FileZilla, Cyberduck, etc.)
- Upload `index.html` and `styles.css` to your public_html or www directory
- Ensure `index.html` is in the root web directory

**2. cPanel File Manager:**
- Log into cPanel
- Open File Manager
- Navigate to public_html (or www)
- Upload `index.html` and `styles.css`

**3. Git Deployment (if using GitHub Pages, Vercel, Netlify, etc.):**
- Push files to your repository
- Configure your hosting to deploy from the repo

**4. Command Line (SSH):**
```bash
# If you have SSH access to your server
scp index.html styles.css user@your-server:/path/to/public_html/
```

After uploading, visit https://erised.me to verify it's working.

