"""Bounded browser geolocation request used by the Streamlit UI."""

GEOLOCATION_JS = """
(() => new Promise((resolve) => {
    let settled = false;
    const finish = (result) => {
        if (settled) return;
        settled = true;
        clearTimeout(deadline);
        resolve(result);
    };
    // The browser's timeout does not cover waiting for permission.
    const deadline = setTimeout(() => finish({error: {code: 'deadline'}}), 15000);
    if (!window.isSecureContext) {
        finish({error: {code: 'insecure'}});
        return;
    }
    if (!navigator.geolocation) {
        finish({error: {code: 'unsupported'}});
        return;
    }
    try {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const {latitude, longitude, accuracy} = position.coords;
                if (!Number.isFinite(latitude) || !Number.isFinite(longitude)
                    || Math.abs(latitude) > 90 || Math.abs(longitude) > 180) {
                    finish({error: {code: 'invalid'}});
                    return;
                }
                // Only rounded coordinates cross the component boundary.
                finish({coords: {
                    latitude: Math.round(latitude * 1000) / 1000,
                    longitude: Math.round(longitude * 1000) / 1000,
                    accuracy: Number.isFinite(accuracy) ? accuracy : null,
                }});
            },
            (error) => finish({error: {code: error.code}}),
            {enableHighAccuracy: false, timeout: 10000, maximumAge: 60000},
        );
    } catch (error) {
        finish({error: {code: error.name === 'SecurityError' ? 1 : 'unavailable'}});
    }
}))()
"""

GEOLOCATION_ERRORS = {
    "1": "Доступ к геолокации запрещён. Разрешите местоположение в настройках сайта "
         "рядом с адресной строкой и проверьте разрешение браузера в настройках геолокации системы.",
    "2": "Браузер не смог определить координаты. Проверьте интернет и службы геолокации системы.",
    "3": "Браузер не определил координаты за 10 секунд.",
    "deadline": "За 15 секунд координаты не получены. Проверьте запрос разрешения рядом "
                "с адресной строкой и службы геолокации системы.",
    "insecure": "Для геолокации откройте приложение через HTTPS или локально на localhost / 127.0.0.1.",
    "unsupported": "Этот браузер не поддерживает геолокацию.",
}


def geolocation_error(code: object) -> str:
    hint = GEOLOCATION_ERRORS.get(str(code), "Браузер не смог передать корректные координаты.")
    return hint + " Повторите попытку или выберите точку старта вручную."
