## instalación

    git clone https://github.com/jbianchi81/pronos_delta.git pronos_delta
    cd pronos_delta
    python3 -m venv .
    source bin/activate
    pip3 install -r requirements.txt

## configuración

    cp config_empty.json config.json
    nano config.json
    # completar parámetros de conexión a api y a base de datos

## ejecución

    python3 pronos_delta.py

