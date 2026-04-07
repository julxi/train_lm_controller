# deploy.py
from utils import (
    ensure_ssh_config,
    get_git_identity,
    get_vast_instance_details,
    load_settings,
    make_connection,
    remote_quote,
)


settings = load_settings()

ip_addr, port = get_vast_instance_details(settings)
config_path = ensure_ssh_config(settings, ip_addr, port)
print(f"SSH configuration written to {config_path}")

git_user_name, git_user_email = get_git_identity()

repo_dir = settings.remote_repo_dir
requirements = f"{repo_dir}/requirements.txt"

with make_connection(settings) as conn:
    conn.run(f"mkdir -p {remote_quote(settings.remote_workdir)}")

    # Fail loudly unless this is genuinely an expected probe.
    conn.run(
        (
            f"if [ -d {remote_quote(repo_dir)}/.git ]; then "
            f"cd {remote_quote(repo_dir)} && git pull --rebase; "
            f"else git clone {remote_quote(settings.repo_url)} {remote_quote(repo_dir)}; "
            f"fi"
        )
    )

    conn.run(
        f"uv pip install --python {remote_quote(settings.remote_python)} "
        f"-r {remote_quote(requirements)}"
    )

    conn.run(f"git config --global user.name {remote_quote(git_user_name)}")
    conn.run(f"git config --global user.email {remote_quote(git_user_email)}")
