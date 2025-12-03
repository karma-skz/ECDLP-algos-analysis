## App Demo

### Modules to install : 
- eventlet
- flask 
- flask-socketio 
- scapy 
- requests 
- cryptography


### How to use

```sh

# for first time (not required every time)
make venv

# in root folder
make   # to start server

# Go to 127.0.0.1:5500 and start capture

# Open two terminals and run :
make alice
make bob

# Select the handshake packets and try attacking

# Use the key deciphered to decrypt the message
```