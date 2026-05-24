module.exports = {
  apps: [{
    name: 'pinku',
    script: '/Users/vivekladha/pinku/.venv/bin/python3',
    args: 'pinku.py',
    cwd: '/Users/vivekladha/pinku',
    interpreter: 'none',          // don't wrap in node — run python3 directly
    autorestart: true,
    restart_delay: 5000,          // wait 5s before restarting on crash
    max_restarts: 10,
    env: {
      PATH: '/Users/vivekladha/pinku/.venv/bin:/usr/local/bin:/usr/bin:/bin',
      VIRTUAL_ENV: '/Users/vivekladha/pinku/.venv',
    }
  }]
}
