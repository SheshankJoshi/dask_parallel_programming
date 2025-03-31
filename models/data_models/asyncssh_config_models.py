from pydantic import BaseModel, Field, FilePath, field_validator
from typing import Optional, Union, List, Tuple
import socket
from asyncssh.connection import SSHClientConnection, SSHClientConnectionOptions

class AsyncSSHConnectOptions(BaseModel):
    host: Optional[str] = Field(
        None,
        description="The hostname or address to connect to."
    )
    port: Optional[int] = Field(
        None,
        description="The port number to connect to. Defaults to the SSH port if not specified."
    )
    tunnel: Optional[Union[str, SSHClientConnection]] = Field(
        None,
        description=(
            "An existing SSH client connection or a string of the form [user@]host[:port] "
            "to tunnel the connection through. A comma-separated list may also be specified."
        )
    )
    family: Optional[int] = Field(
        socket.AF_UNSPEC,
        description=(
            "The address family to use when creating the socket. "
            "Defaults to automatic selection based on the host."
        )
    )
    flags: Optional[int] = Field(
        None,
        description="Flags to pass to getaddrinfo() when looking up the host address."
    )
    local_addr: Optional[Tuple[str, int]] = Field(
        None,
        description="The host and port to bind the socket to before connecting."
    )
    sock: Optional[socket.socket] = Field(
        None,
        description=(
            "An existing already-connected socket to run SSH over, instead of opening a new connection. "
            "If specified, host, port, family, flags, or local_addr should not be specified."
        )
    )
    config: Optional[Union[None, List[FilePath]]] = Field(
        None,
        description=(
            "Paths to OpenSSH client configuration files to load. If not specified, defaults to .ssh/config. "
            "If explicitly set to None, no new configuration files will be loaded."
        )
    )
    #! NOTE : We have to set the command here to activate an environment on shell creation. For that we need to set SSH Connection Options here.
    # TODO : This needs to be properly studied and configured so that we can proper environment activated before anything else happens
    # For e.g. import asyncssh
    # options = asyncssh.SSHClientConnectionOptions(
    #     host='remote.example.com',
    #     command='conda activate env_name' # or bash -c 'conda activate env_name && exec bash
    # )
    options: Optional[SSHClientConnectionOptions] = Field(
        None,
        description=(
            "Options to use when establishing the SSH client connection. "
            "These can be specified either here or as direct keyword arguments."
        )
    )

    @field_validator("port")
    def validate_port(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and (value < 1 or value > 65535):
            raise ValueError("Port must be between 1 and 65535.")
        return value

    @field_validator("local_addr")
    def validate_local_addr(cls, value: Optional[Tuple[str, int]]) -> Optional[Tuple[str, int]]:
        if value:
            host, port = value
            if not isinstance(host, str):
                raise ValueError("local_addr host must be a string.")
            if not isinstance(port, int) or not (1 <= port <= 65535):
                raise ValueError("local_addr port must be an integer between 1 and 65535.")
        return value

    @field_validator("family")
    def validate_family(cls, value: Optional[int]) -> Optional[int]:
        if value not in {socket.AF_UNSPEC, socket.AF_INET, socket.AF_INET6}:
            raise ValueError("family must be one of socket.AF_UNSPEC, socket.AF_INET, or socket.AF_INET6.")
        return value

    @field_validator("config", mode="before")
    def validate_config(cls, value: Optional[Union[None, List[FilePath]]]) -> Optional[Union[None, List[FilePath]]]:
        if value is not None and not isinstance(value, list):
            raise ValueError("config must be a list of file paths or None.")
        return value


# Example usage
if __name__ == "__main__":
    try:
        ssh_options = AsyncSSHConnectOptions(
            host="example.com",
            port=22,
            family=socket.AF_INET,
            local_addr=("0.0.0.0", 0),
            config=["/home/user/.ssh/config"],
        )
        print(ssh_options.model_dump())
    except ValueError as e:
        print(f"Validation error: {e}")