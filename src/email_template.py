# src/email_template.py
from jinja2 import Template

# A simple template with placeholders
TEMPLATES = {
    'A': """Subject: {{ subject }}

Hi {{ first_name }},

We thought you'd like this — specially picked for someone {{ age }} years young in {{ location }}.

{{ body }}

Cheers,
Your Company
""",
    'B': """Subject: {{ subject }}

Hey {{ first_name }},

Quick one — we picked something you might love based on your last visits.

{{ body }}

Best,
Team
"""
}

def render_template(variant, context):
    if variant not in TEMPLATES:
        raise ValueError("Invalid variant")
    tmpl = Template(TEMPLATES[variant])
    return tmpl.render(**context)

# Example helper to build context
def build_context(row):
    # row: pandas Series with 'email' and 'age', 'location'
    first_name = row.get('email', '').split('@')[0]
    return {
        'first_name': first_name.capitalize(),
        'age': int(row.get('age',0)),
        'location': row.get('location',''),
        'subject': 'A special pick for you',
        'body': 'Check out our new offers tailored to your interests!'
    }
