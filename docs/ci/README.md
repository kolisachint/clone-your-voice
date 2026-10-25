# CI workflow (pending activation)

`ci.yml` here is the project's GitHub Actions workflow. It lives outside `.github/workflows/`
only because the automation that pushed it lacked GitHub's `workflow` permission.

To activate it, a maintainer moves it into place (locally or via the GitHub web editor):

```sh
git mv docs/ci/ci.yml .github/workflows/ci.yml && git commit -m "ci: activate workflow" && git push
```
