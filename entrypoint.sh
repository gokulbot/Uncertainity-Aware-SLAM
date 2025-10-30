#!/bin/bash
set -e

USER_ID=${LOCAL_USER_ID:-1000}
GROUP_ID=${LOCAL_GROUP_ID:-1000}
USER_NAME=${LOCAL_USER_NAME:-developer}

if ! id "$USER_NAME" &>/dev/null; then
    groupadd -g "$GROUP_ID" "$USER_NAME"
    useradd -m -s /bin/bash -u "$USER_ID" -g "$GROUP_ID" "$USER_NAME"
fi

echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

export HOME=/home/$USER_NAME
exec gosu "$USER_NAME" "$@"
        