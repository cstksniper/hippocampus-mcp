# -*- coding: utf-8 -*-
"""Versioned hook shim sources (naturalization P0-1, 2026-11-07).

The scripts in this package are TEMPLATES: ``@@TOKEN@@`` placeholders are
substituted by :mod:`hippocampus_mcp.installer` at deploy time (repo src
path, embedding model, user-library path). Before P0 the deployed shims
were hand-placed one-offs with no versioned source — reinstalling after a
host upgrade meant archaeology. Now deploy = copy + substitute, and
reinstall = repair.

These files are NOT importable as-is (tokens are invalid Python literals)
and are never imported by the server; they are data for the installer and
doctor's drift check.
"""
