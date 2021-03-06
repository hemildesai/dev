var scheme = "light";
var savedScheme = localStorage.getItem("scheme");

var container = document.getElementsByTagName("html")[0];
var prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

if (prefersDark) {
  scheme = "dark";
}

if (savedScheme) {
  scheme = savedScheme;
}

if (scheme == "dark") {
  darkscheme(undefined, container);
} else {
  lightscheme(undefined, container);
}

$(window).ready(function () {
  var toggle = document.getElementById("scheme-toggle");

  if (scheme == "dark") {
    darkscheme(toggle, container);
  } else {
    lightscheme(toggle, container);
  }

  toggle.addEventListener("click", () => {
    if (toggle.className === "light") {
      darkscheme(toggle, container);
    } else if (toggle.className === "dark") {
      lightscheme(toggle, container);
    }
  });

  var btn = $("#scroll-to-top");

  $(window).scroll(function () {
    if ($(window).scrollTop() > 300) {
      btn.addClass("show");
    } else {
      btn.removeClass("show");
    }
  });

  btn.on("click", function (e) {
    e.preventDefault();
    $("html, body").animate({ scrollTop: 0 }, "100");
  });
});

function darkscheme(toggle, container) {
  localStorage.setItem("scheme", "dark");
  if (toggle) {
    toggle.innerHTML = feather.icons.sun.toSvg();
    toggle.className = "dark";
  }
  container.className = "dark";
}

function lightscheme(toggle, container) {
  localStorage.setItem("scheme", "light");
  if (toggle) {
    toggle.innerHTML = feather.icons.moon.toSvg();
    toggle.className = "light";
  }
  container.className = "";
}
