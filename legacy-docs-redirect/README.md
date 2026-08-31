# Legacy docs domain redirect

The documentation moved to `gpjax.quantclimate.com` (GitHub Pages). This
directory is the entire contents of the Netlify site that keeps the **old**
host, `docs.jaxgaussianprocesses.com`, alive as a permanent redirect.

It is not part of the docs build and is not deployed by CI. It is deployed by
hand, once, and then only again if the redirect target changes.

## Why this exists

GitHub Pages serves exactly one custom domain per site, so it cannot answer for
both hosts. Without this, every published link to the old domain 404s — the JOSS
paper, the PyPI project page, search results, and any third-party citation.

## Deploying it

> [!WARNING]
> Deploy this to the Netlify site that serves **`docs.jaxgaussianprocesses.com`**.
> That is a different site from the one serving the `jaxgaussianprocesses.com`
> apex and `www` (the marketing site). Deploying this bundle to the marketing
> site would replace it with a redirect to the GPJax docs.
>
> Confirm before deploying — `netlify sites:list`, then check which site claims
> the `docs.` subdomain under Domain management.

The docs site is the same one this repo already uses for PR previews, i.e. the
site behind the `NETLIFY_SITE_ID` repository secret.

```bash
npx --yes netlify-cli@latest deploy \
  --prod \
  --dir=legacy-docs-redirect \
  --site="<site-id>" \
  --message="301 legacy docs host to gpjax.quantclimate.com"
```

`--prod` is correct **here** and only here: this is a deliberate, manual
replacement of that site's production deploy. It must never appear in
`.github/workflows/test_docs.yml`, whose preview step uses `--alias` so a pull
request can never overwrite a production deploy.

Deploying this does not affect PR previews. Those are alias deploys at
`pr-<n>--<site>.netlify.app` and are independent of the production deploy.

## Verifying

```bash
# Expect: 301, and a Location on the new host with the path preserved.
curl -sI https://docs.jaxgaussianprocesses.com/examples/regression.html \
  | grep -iE '^HTTP|^location'

# Expect: the same, through an old MkDocs-era URL. The new site's reredirect
# stub then completes the hop to /examples/classification.html.
curl -sI https://docs.jaxgaussianprocesses.com/_examples/classification/ \
  | grep -iE '^HTTP|^location'
```
