
/*
* Popup Modal Ribbon
* ========================================================================== */
let hide;
(function() {
  hide = function() {
    var ribbon = document.getElementsByClassName('essential-math-ribbon')[0];
    if (ribbon) {
      ribbon.style.display = 'none';
      document.cookie = "essential-math-ribbon=1; path=/ ; expires= Thu, 21 Aug 2040 20:00:00 UTC; SameSite=Strict";
    }
  }
})();

(function() {
  console.log('asdf');
  function getCookie(cname) {
    var name = cname + "=";
    var decodedCookie = decodeURIComponent(document.cookie);
    var ca = decodedCookie.split(';');
    for(var i = 0; i <ca.length; i++) {
      var c = ca[i];
      while (c.charAt(0) == ' ') {
        c = c.substring(1);
      }
      if (c.indexOf(name) == 0) {
        return c.substring(name.length, c.length);
      }
    }
    return "";
  }

  const url = window.location.href;
  if (url.split('/').includes('Essential-Math-for-Data-Science')) {
    document.cookie = "essential-math-ribbon=1; path=/ ; expires= Thu, 21 Aug 2040 20:00:00 UTC; SameSite=Strict";
  }

  setTimeout(() => {
    if (getCookie('essential-math-ribbon') !== "1") {
      var ribbon = document.getElementsByClassName('essential-math-ribbon')[0];
      if (ribbon) {
        ribbon.style.display = 'flex';
        document.cookie = "essential-math-ribbon=1; path=/ ; expires= Thu, 21 Aug 2040 20:00:00 UTC; SameSite=Strict";
      }
    }
  }, 3000);

})();
