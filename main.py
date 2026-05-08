from app.core.config import get_settings
from app.core.logger import app_logger

settings = get_settings()

app_logger.info(f"Starting {settings.app_name}")
app_logger.info(f"Environment: {settings.app_env}")
app_logger.info(f"Log Level: {settings.log_level}")
app_logger.debug("Config and Logger are working correctly!")