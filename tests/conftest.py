"""Shared test config: enable JAX x64 so analytic references are precise."""

import jax

jax.config.update("jax_enable_x64", True)
