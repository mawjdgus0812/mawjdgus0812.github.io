---
layout: default
title: Archive2
use_math: true
---

# Archive2

Browse all posts by month and year.

{% assign postsByYearMonth = site.posts | group_by_exp: "post2", "post2.date | date: '%B %Y'" %}
{% for yearMonth in postsByYearMonth %}
  <h2>{{ yearMonth.name }}</h2>
  <ul>
    {% for post in yearMonth.items %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}
