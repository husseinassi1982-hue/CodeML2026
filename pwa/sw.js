/* DayOne service worker: keeps the chat page on the phone so it opens with no network.
   Only the app shell is cached. API calls (photos, fields, patients) always go to the
   server and are never cached here: photos waiting for the network live in the page's
   encrypted outbox (IndexedDB), not in this cache. */
const CACHE = "dayone-shell-v1";
const SHELL = ["/", "/manifest.webmanifest", "/icon.svg"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))));
  self.clients.claim();
});

// Network first (so a new version of the page arrives when online), cache when offline.
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin || !SHELL.includes(url.pathname)) return;
  const timeout = new Promise((_, reject) => setTimeout(() => reject(new Error("slow network")), 4000));
  e.respondWith(
    Promise.race([fetch(e.request), timeout])
      .then(res => {
        if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(url.pathname, copy)); }
        return res;
      })
      .catch(() => caches.match(url.pathname).then(hit => hit || Response.error()))
  );
});
