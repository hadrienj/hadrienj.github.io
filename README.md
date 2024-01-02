# Creation of the blog

This blog was created using Jekyll (with the help of [this theme](https://github.com/redVi/voyager)) and github pages. You can find it [here](https://hadrienj.github.io/).

# Cookies

A cookie consent banner can be used if necessary with `{% include cookie-consent.html %}`. Details on what information is stored is explained in the page `/privacy`.

# Hotjar

Hotjar service has been disable. It can be enabled by uncommenting these lines:

```html
<!-- <script>
      (function(h,o,t,j,a,r){
          h.hj=h.hj||function(){(h.hj.q=h.hj.q||[]).push(arguments)};
          h._hjSettings={hjid:1555723,hjsv:6};
          a=o.getElementsByTagName('head')[0];
          r=o.createElement('script');r.async=1;
          r.src=t+h._hjSettings.hjid+j+h._hjSettings.hjsv;
          a.appendChild(r);
      })(window,document,'https://static.hotjar.com/c/hotjar-','.js?sv=');
  </script> -->
```

in the file `head.html`.
