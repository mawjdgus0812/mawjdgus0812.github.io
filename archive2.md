---
layout: default
title: Code
use_math: true
---

# Code review

Browse all posts by month and year.


{% assign postsByYearMonth = site.posts | group_by_exp: "post", "post.date | date: '%B %Y'" %}
{% for yearMonth in postsByYearMonth %}
  <h2>{{ yearMonth.name }}</h2>
  <ul>
    {% for post in yearMonth.items %}
      {%-if post.category == "code"-%}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
      {%-endif-%}
    {% endfor %}
  </ul>
{% endfor %}
