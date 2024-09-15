class DataException(Exception):
    """Excepción personalizada para errores en la carga de datos."""
    def __init__(self, message):
        super().__init__(message)
