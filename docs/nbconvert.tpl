{% extends 'markdown.tpl' %}

<!-- ignore cell with "# ignore"  -->
{% block input %}
{% if "# ignore" not in cell.source %}
```
{%- if 'magics_language' in cell.metadata  -%}
    {{ cell.metadata.magics_language}}
{%- elif 'name' in nb.metadata.get('language_info', {}) -%}
    {{ nb.metadata.language_info.name }}
{%- endif %}
{{ cell.source.replace(";", "") }}
```
{% else %}
{% endif %}
{% endblock input %}

<!-- fix output format -->
{% block stream %}
<pre class="nb-output"><code> >> {{ output.text }}</code></pre>
{% endblock stream %}

{% block data_text scoped %}
<pre class="nb-output"><code> >> {{ output.data['text/plain'] }}</code></pre>
{% endblock data_text %}

{% block data_html scoped %}
<pre class="nb-output"><code>{{ output.data["text/html"] }}</code></pre>
{% endblock data_html %}
