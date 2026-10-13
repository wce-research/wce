# Repository publication policy

Public WCE publishes code, protocols, positive/negative/inconclusive research results,
portable runtime behavior, frontend/branding and reproducibility instructions.
Tenant addresses, operator identity/network information, credentials and internal
production runbooks are excluded. Deployment-oriented code is not a dependency of
the scientific engine or local frontend. No production changes accompany repository
maintenance.

Before publishing run `make repository-audit-bootstrap repository-audit AUDIT_STRICT=1`.
The scanner inspects every reachable Git object/commit/tag plus current tracked and
relevant untracked files, uses Gitleaks with redacted decoded/archive scanning, and
keeps reports outside Git. Supply an external AUDIT_POLICY for known prohibited
identifiers; never embed them in tests/configuration. Review generic IP/account hits:
documentation/test addresses and public AWS image-publisher IDs are not credentials.

Sensitive-data cleanup may change commit IDs. Collaborators must obtain a fresh
clone after notified history sanitization and must not merge old unsanitized refs.
Scientific artifacts remain intact. Git-history deletion is not credential revocation
and cannot erase copies retained by third parties. See SECURITY.md for disclosure.
