function cookieConsentCreateCookie(name,value,days) {
    var expires = "";
    if (days) {
        var date = new Date();
        date.setTime(date.getTime() + (days*24*60*60*1000));
        expires = "; expires=" + date.toUTCString();
    }
    document.cookie = name + "=" + value + expires + "; path=/";
}

function cookieConsentReadCookie(name) {
    var nameEQ = name + "=";
    var ca = document.cookie.split(';');
    for(var i=0;i < ca.length;i++) {
        var c = ca[i];
        while (c.charAt(0)==' ') c = c.substring(1,c.length);
        if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length,c.length);
    }
    return null;
}

// Grant consent immediately if already approved
if(cookieConsentReadCookie('cookie-notice-dismissed')=='true') {
    if (typeof gtag !== 'undefined') {
        gtag('consent', 'update', {
            'analytics_storage': 'granted'
        });
    }
}

// Main initialization function
function initCookieBanner() {
    if(cookieConsentReadCookie('cookie-notice-dismissed')=='true') {
        setTimeout(() => {
            var article = document.getElementsByTagName("article")[0];
            if (article) article.style.padding = "2em 1em";
        }, 500);
    } else {
        var cookieNotice = document.getElementById('cookie-notice');
        if (cookieNotice) {
            cookieNotice.style.display = 'block';
        }
        setTimeout(() => {
            var article = document.getElementsByTagName("article")[0];
            if (article) article.style.padding = "5em 1em";
        }, 500);
    }

    var acceptButton = document.getElementById('cookie-notice-accept');
    if (acceptButton) {
        acceptButton.addEventListener("click", function() {
            cookieConsentCreateCookie('cookie-notice-dismissed','true',31);
            var cookieNotice = document.getElementById('cookie-notice');
            if (cookieNotice) cookieNotice.style.display = 'none';

            // Update Google Analytics consent
            if (typeof gtag !== 'undefined') {
                gtag('consent', 'update', {
                    'analytics_storage': 'granted'
                });
            }

            location.reload();
        });
    }
}

// Execute immediately if DOM is ready, otherwise wait
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCookieBanner);
} else {
    initCookieBanner();
}
