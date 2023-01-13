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



document.addEventListener('DOMContentLoaded', function() {
  var elems = document.querySelectorAll('.collapsible');
  console.log(elems)
  var instances = M.Collapsible.init(elems, {
    accordion: false
  });
});


const eventTag = document.querySelector('.eventTag')
eventTag.addEventListener('click', handleClickEvent, false);

function handleClickEvent(e) {
  const url = e.view.location.href;
  if (document.location.hostname.search("hadrienj.github.io") !== -1) {
    gtag('event', 'click', {
      'event_category': 'click',
      'event_label': `to_essential_math_from_${url}`,
      'value': 1
    });
    }
}

document.addEventListener('DOMContentLoaded', function() {
  var elems = document.querySelectorAll('.collapsible');
  console.log(elems)
  var instances = M.Collapsible.init(elems, {
    accordion: false
  });
});


// Cookies
function createCookie(name, value, days) {
  if (days) {
      var date = new Date();
      date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
      var expires = "; expires=" + date.toGMTString();
  }
  else var expires = "";               
  console.log("create")
  document.cookie = name + "=" + value + expires + "; path=/";
}

function readCookie(name) {
  var nameEQ = name + "=";
  var ca = document.cookie.split(';');
  for (var i = 0; i < ca.length; i++) {
      var c = ca[i];
      while (c.charAt(0) == ' ') c = c.substring(1, c.length);
      if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length, c.length);
  }
  return null;
}

function eraseCookie(name) {
  createCookie(name, "", -1);
}

function collapseEssentialMathRibbon() {
  createCookie("essentialMathAlreadySeen", 1, days=10);
  $('.card-section-ribbon').css({
    'transform': 'none',
    'top': 'inherit',
    'bottom': '0',
    'left': '0',
    'right': '',
    'margin': '0',
    'justify-content': 'center',
    'width': '100%',
    'padding': '0.5em 0 0 0',
    'border-radius': '0'
  });
  $('.card-section-ribbon-img').css({'flex': '0 0 5%', 'text-align': 'center', 'display': 'none'});
  $('.essential-math-text').css({'flex': '0 0 30%', 'display': 'none'});
  $('.button-get-the-book').css({'font-size': '1rem'}).html("<b>GET THE BOOK</b><br>Essential Math for Data Science");

  $('.offer').css({
      'display': 'none'
  });
  $('.get-book').css({
      'display': 'none'
  });
  $('.content-wrapper,.sidebar').css({
      'filter': 'blur(0px)'
  });
  $('.more-math').css({
    'display': 'none'
  });
};

// function expandEssentialMathRibbon() {
//   $('.card-section-ribbon').css({
//     'transform': 'translate(-50%, -50%)',
//     'top': '50%',
//     // 'bottom': '0',
//     'left': '50%',
//     'right': '-20%',
//     // 'margin': '0',
//     // 'justify-content': 'center',
//     // 'width': '100%',
//     // 'padding': '0.5em 0 0 0',
//     'border-radius': '13px'
//   });
//   $('.card-section-ribbon-img').css({'flex': '0 0 5%', 'text-align': 'center', 'display': 'none'});
//   $('.essential-math-text').css({'flex': '0 0 30%', 'display': 'none'});
//   $('.button-get-the-book').css({'font-size': '1rem'}).html("<b>GET THE BOOK</b><br>Essential Math for Data Science");

//   $('.offer').css({
//       'display': 'none'
//   });
//   $('.get-book').css({
//       'display': 'none'
//   });
//   $('.content-wrapper,.sidebar').css({
//       'filter': 'blur(0px)'
//   });
//   $('.more-math').css({
//     'display': 'none'
//   });
// };

$(document).ready(function(){
  $( ".collapsible-header" ).click(function() {
      $(".more",this).toggle()
      $(".less", this).toggle()
  });

  let essentialMathAlreadySeen = readCookie("essentialMathAlreadySeen");
  console.log("essentialMathAlreadySeen", essentialMathAlreadySeen)
  if (essentialMathAlreadySeen === '1') {
    collapseEssentialMathRibbon();
  } else {
    $('.content-wrapper,.sidebar').css({
      'filter': 'blur(5px)'
    })

  }
  $('.card-section-ribbon').addClass('show-inline-flex');

  $('.content-wrapper,.sidebar').click(collapseEssentialMathRibbon);

});


if (document.location.hostname.search("www.essentialmathfordatascience.com") !== -1) {
      gtag('event', 'click', {
        'event_category': 'get_product',
        'event_label': 'product',
        'value': product
      });
    }
