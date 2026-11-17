/* Site-wide behaviour: mobile navigation and FAQ reference filtering.
   User and FAQ text is only ever written with textContent / attributes. */
(function () {
  'use strict';

  // ---- mobile navigation ----
  var toggle = document.querySelector('.nav-toggle');
  var nav = document.getElementById('site-nav');
  if (toggle && nav) {
    toggle.addEventListener('click', function () {
      var open = nav.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && nav.classList.contains('is-open')) {
        nav.classList.remove('is-open');
        toggle.setAttribute('aria-expanded', 'false');
        toggle.focus();
      }
    });
  }

  // ---- FAQ reference search and category filter ----
  var list = document.getElementById('faq-list');
  if (!list) { return; }

  var searchInput = document.getElementById('faq-search');
  var categorySelect = document.getElementById('faq-category');
  var countLabel = document.getElementById('faq-count');
  var emptyState = document.getElementById('faq-empty');
  var cards = Array.prototype.slice.call(list.querySelectorAll('.faq-card'));
  var total = cards.length;

  function applyFilters() {
    var terms = (searchInput.value || '').toLowerCase().split(/\s+/).filter(Boolean);
    var category = categorySelect.value;
    var visible = 0;

    cards.forEach(function (card) {
      var haystack = card.getAttribute('data-search') || '';
      var matchesText = terms.every(function (term) { return haystack.indexOf(term) !== -1; });
      var matchesCategory = !category || card.getAttribute('data-category') === category;
      var show = matchesText && matchesCategory;
      card.hidden = !show;
      if (show) { visible += 1; }
    });

    countLabel.textContent = 'Showing ' + visible + ' of ' + total + ' FAQs';
    emptyState.hidden = visible !== 0;
    list.hidden = visible === 0;
  }

  searchInput.addEventListener('input', applyFilters);
  categorySelect.addEventListener('change', applyFilters);
  applyFilters();
})();
