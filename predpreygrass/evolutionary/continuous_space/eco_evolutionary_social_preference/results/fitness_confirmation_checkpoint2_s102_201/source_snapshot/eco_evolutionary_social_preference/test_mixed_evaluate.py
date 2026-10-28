from .mixed_evaluate import canonical


def test_spell_order_is_ignored_but_real_content_and_field_order_are_not():
    a={'spell_records':[[4,'death',False,True],[2,'separation',False,False]],'record_fields':['duration','reason']}
    b={'spell_records':[list(row) for row in reversed(a['spell_records'])],'record_fields':list(a['record_fields'])}
    assert canonical(a)==canonical(b)
    b['spell_records'][0][0]=3
    assert canonical(a)!=canonical(b)
    assert canonical({'record_fields':['duration','reason']})!=canonical({'record_fields':['reason','duration']})
