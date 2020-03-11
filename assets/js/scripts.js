/*
* Zoom Images, Get Date and Shadow
* ========================================================================== */

(function() {
  /* variables */
  var shadow = document.getElementById('shadow');
  var images = document.querySelectorAll('a img');
  var imageHeight = window.innerHeight - 20;

  /* events */
  shadow.addEventListener('click', resetShadow, false);
  window.addEventListener('keydown', resetStyles, false);
  window.addEventListener('resize', refreshImageSizes, false);

  /* functions */
  setDate();
  // toggleImages();


  function setDate() {
    var currentYear = document.querySelector('.full-year');
    if (currentYear) {
      currentYear.innerHTML = new Date().getFullYear();
    }
  }

  function refreshImageSizes() {
    // select all images
    [].forEach.call(images, function(img) {
      // if image zoomed
      if (img.classList.contains('img-popup')) {
        img.style.maxHeight = imageHeight + 'px';
        img.style.marginLeft = '-' + (img.offsetWidth / 2) + 'px';
        img.style.marginTop = '-' + (img.offsetHeight / 2) + 'px';
      }
    });
  }

  function resetShadow() {
    shadow.style.display = 'none';
    resetAllImages();
  }

  function resetStyles(event) {
    if (event.keyCode == 27) {
      event.preventDefault();
      shadow.style.display = 'none';
      resetAllImages();
    }
  }

  function resetAllImages() {
    [].forEach.call(images, function(img) {
      img.classList.remove('img-popup');
      img.style.cursor = 'zoom-in';
      img.style.marginLeft = 'auto';
      img.style.marginTop = 'auto';
    });
  }

  function toggleImages() {
    [].forEach.call(images, function(img) {
      img.addEventListener('click', function(event) {
        event.preventDefault();
        img.classList.toggle('img-popup');
        if (img.classList.contains('img-popup')) {
          img.style.cursor = 'zoom-out';
          img.style.maxHeight = imageHeight + 'px';
          img.style.marginLeft = '-' + (img.offsetWidth / 2) + 'px';
          img.style.marginTop = '-' + (img.offsetHeight / 2) + 'px';
          shadow.style.display = 'block';
        } else {
          img.style.cursor = 'zoom-in';
          img.style.maxHeight = '100%';
          img.style.marginLeft = 'auto';
          img.style.marginTop = 'auto';
          shadow.style.display = 'none';
        }
      });
    });
  }
})();


/*
* Aside Resize
* ========================================================================== */

(function() {
  var aside = document.querySelector('.sidebar');
  var mainContainer = document.querySelectorAll('.content-wrapper');
  var switcher = document.getElementById('switcher');

  switcher.addEventListener('click', slide, false);


  function slide() {
    aside.classList.add('transition-divs');
    aside.classList.toggle('aside-left');
    [].forEach.call(mainContainer, function(c) {
      c.classList.add('transition-divs');
      c.classList.toggle('centering');
    });
  }
})();



/*
* Popup Modal Ribbon
* ========================================================================== */
let hide;
(function() {
  hide = function() {
    document.getElementsByClassName('essential-math-ribbon')[0].style.display = 'none'; //gets the element and sets display to none
    document.cookie = "essential-math-ribbon=1; path=/ ; expires= Thu, 21 Aug 2020 20:00:00 UTC; SameSite=Strict";
  }
})();

// Check cookies
(function() {
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
    document.cookie = "essential-math-ribbon=1; path=/ ; expires= Thu, 21 Aug 2020 20:00:00 UTC; SameSite=Strict";
  }

  setTimeout(() => {
    if (getCookie('essential-math-ribbon') !== "1") {
      console.log('asdf', document.cookie)
      document.getElementsByClassName('essential-math-ribbon')[0].style.display = 'flex';
      document.cookie = "essential-math-ribbon=1; path=/ ; expires= Thu, 21 Aug 2020 20:00:00 UTC; SameSite=Strict";
    }
  }, 3000);

})();
