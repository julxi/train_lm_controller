# utils.py
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple
import os
import shlex

import paramiko
from dotenv import load_dotenv
from fabric import Connection
from git import Repo
from vastai_sdk import VastAI


ENV_FILES = (".env.publish", "vastai/.env")


@dataclass(frozen=True)
class Settings:
    ssh_config_dir: Path
    ssh_config_name: str
    ssh_identity_file: Path
    ssh_user: str
    ssh_host_alias: str

    repo_url: str
    repo_name: str

    remote_python: str
    remote_workdir: str

    vastai_api_key: str
    vastai_instance_id: str

    @property
    def ssh_config_file(self) -> Path:
        return self.ssh_config_dir / self.ssh_config_name

    @property
    def remote_repo_dir(self) -> str:
        # remote Linux path, so keep it as a POSIX-style string.
        return f"{self.remote_workdir.rstrip('/')}/{self.repo_name}"


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    for env_file in ENV_FILES:
        load_dotenv(env_file)

    return Settings(
        ssh_config_dir=Path(_required_env("SSH_CONFIG_DIR")).expanduser(),
        ssh_config_name=_required_env("SSH_CONFIG_NAME"),
        ssh_identity_file=Path(_required_env("SSH_IDENTITY_FILE")).expanduser(),
        ssh_user=_required_env("SSH_USER"),
        ssh_host_alias=_required_env("SSH_HOST_ALIAS"),
        repo_url=_required_env("REPO_URL"),
        repo_name=_required_env("REPO_NAME"),
        remote_python=_required_env("REMOTE_PYTHON"),
        remote_workdir=_required_env("REMOTE_WORKDIR"),
        vastai_api_key=_required_env("VASTAI_API_KEY"),
        vastai_instance_id=_required_env("VASTAI_INSTANCE_ID"),
    )


def get_vast_instance_details(settings: Settings) -> tuple[str, int]:
    vast_sdk = VastAI(api_key=settings.vastai_api_key)
    instance = vast_sdk.show_instance(id=settings.vastai_instance_id)

    if not instance:
        raise RuntimeError(
            f"No Vast.ai instance found for id {settings.vastai_instance_id}"
        )

    try:
        ip_addr = instance["public_ipaddr"]
        port = int(instance["ports"]["22/tcp"][0]["HostPort"])
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise RuntimeError(
            "Could not read SSH host/port from Vast.ai instance metadata"
        ) from e

    return ip_addr, port


def ensure_ssh_config(settings: Settings, ip_addr: str, port: int) -> Path:
    settings.ssh_config_dir.mkdir(parents=True, exist_ok=True)

    identity_file = settings.ssh_identity_file.expanduser().resolve()

    content = (
        f"Host {settings.ssh_host_alias}\n"
        f"    HostName {ip_addr}\n"
        f"    User {settings.ssh_user}\n"
        f"    IdentityFile {identity_file.as_posix()}\n"
        f"    Port {port}\n"
    )

    path = settings.ssh_config_file
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current != content:
        path.write_text(content, encoding="utf-8", newline="\n")

    return path


def get_git_identity(repo_path: str | Path = ".") -> tuple[str, str]:
    repo = Repo(repo_path, search_parent_directories=True)

    try:
        reader = repo.config_reader()
        name = str(reader.get_value("user", "name"))
        email = str(reader.get_value("user", "email"))
    except Exception as e:
        raise RuntimeError(
            "Git user.name and user.email are not configured for this repository"
        ) from e

    return name, email


def make_connection(settings: Settings | None = None) -> Connection:
    settings = settings or load_settings()
    ssh_config_file = settings.ssh_config_file

    if not ssh_config_file.exists():
        raise FileNotFoundError(
            f"SSH config file does not exist: {ssh_config_file}. "
            f"Call ensure_ssh_config(...) first."
        )

    ssh_config = paramiko.SSHConfig()
    with ssh_config_file.open("r", encoding="utf-8") as f:
        ssh_config.parse(f)

    host_config = ssh_config.lookup(settings.ssh_host_alias)

    identity = host_config.get(
        "identityfile", [str(settings.ssh_identity_file.expanduser().resolve())]
    )[0]

    return Connection(
        host=host_config["hostname"],
        user=host_config.get("user", settings.ssh_user),
        port=int(host_config.get("port", 22)),
        connect_kwargs={"key_filename": identity},
    )


def remote_quote(value: str) -> str:
    # Handy for shell commands on the remote Linux host.
    return shlex.quote(value)
