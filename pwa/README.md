# pwa/ — the phone's offline app shell

Makes the chat (`front_end2.HTML`) an installable web app that works on a phone with no network.
The backend serves these files from the site root (`backend/main.py`).

| File | What |
|---|---|
| `sw.js` | Service worker: caches the chat page, manifest and icon, so the page opens offline. Network first while online (new versions arrive), cache when offline. API calls are never cached. |
| `manifest.webmanifest` | App name, colours, icon: lets the midwife *Add to Home screen*. |
| `icon.svg` | App icon. |

The photos themselves are not kept here but in the page's **outbox** (`Outbox` in `front_end2.HTML`):

* each photo is encrypted with AES-GCM (WebCrypto) and stored in IndexedDB; the key is generated on the
  phone and cannot be exported;
* when the phone has network, photos are uploaded oldest first (`POST /records` with `captured_at`, the
  time the photo was taken) and deleted from the phone only after the server has answered;
* if the server cannot be reached they stay on the phone and the upload is retried every 20 s;
* the header's **📶 Online** switch simulates losing the network, for demos.

Service workers and WebCrypto need a secure page: `https://…` or `http://localhost`. For a phone on the
same Wi-Fi as the server, see "Phone setup" in the top-level README.
