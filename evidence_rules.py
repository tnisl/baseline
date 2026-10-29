"""Checks on visible tool results only; never reads environment state or labels."""
from __future__ import annotations
import re
from common import result_succeeded


def visible_results(row, observations):
    for turn in row.get('history') or []:
        for call, result in zip(turn.get('tool_calls') or [], turn.get('tool_results') or []):
            if result_succeeded(result):
                yield call.get('name', ''), call.get('arguments') or {}, result
    for item in observations:
        if result_succeeded(item['result']):
            yield item['name'], item.get('arguments') or {}, item['result']


def collection_records(result, collection):
    if not isinstance(result, dict):
        return []
    records = result.get(collection, [])
    if not isinstance(records, list):
        records = [records] if isinstance(records, dict) else []
    return [item for item in records if isinstance(item, dict)]


def reference_candidates(spec, row, observations):
    """Group records by field and x-ref namespace, retaining latest record by ID."""
    output = {}
    for field, prop in (spec.get('parameters', {}).get('properties') or {}).items():
        ref = prop.get('x-ref', '')
        if '.' not in ref:
            continue
        app, collection = ref.split('.', 1)
        records = {}
        for name, args, result in visible_results(row, observations):
            if name.split('.')[0] != app:
                continue
            found = collection_records(result, collection)
            # A successful write/get may return one record rather than a list.
            for key in ('item', collection.rstrip('s')):
                item = result.get(key)
                if (isinstance(item, dict) and item.get('id')
                        and (collection == app or args.get(field) == item['id']
                             or name in prop.get('description', ''))):
                    found.append(item)
            for item in found:
                if item.get('id'):
                    records[item['id']] = item
        output[field] = {'ref': ref, 'records': list(records.values())}
    if spec.get('name') == 'food_delivery.place_order':
        menus = []
        for name, args, result in visible_results(row, observations):
            if name == 'food_delivery.get_menu':
                menus.append({'restaurant_id': args.get('restaurant_id'),
                              'menu': collection_records(result, 'menu')})
        output['items'] = {'ref': 'food_delivery.menu', 'menus': menus}
    return output


def reference_error(name, arguments, spec, request, row, observations):
    candidates = reference_candidates(spec, row, observations)
    for field, info in candidates.items():
        value = arguments.get(field)
        if not isinstance(value, str):
            continue
        records = info.get('records', [])
        record = next((item for item in records if item.get('id') == value), None)
        # Explicit IDs in the current request can be supplied without a lookup.
        if record is None and not re.search(r'(?<![\w])' + re.escape(value) + r'(?![\w])', request):
            return field, f'{field}={value} chưa được xác nhận trong {info["ref"]}; cần tra đúng collection.'
        expected_type = {'smart_home.light_set': 'light', 'smart_home.climate_set': 'climate',
                         'smart_home.fan_set': 'fan'}.get(name)
        if record and expected_type and record.get('type') != expected_type:
            return field, f'{value} có type={record.get("type")}, cần type={expected_type}.'
    if name == 'food_delivery.place_order':
        menu = {}
        for tool, args, result in visible_results(row, observations):
            if tool == 'food_delivery.get_menu' and args.get('restaurant_id') == arguments.get('restaurant_id'):
                menu.update({item['id']: item for item in collection_records(result, 'menu') if item.get('id')})
        for item in arguments.get('items') or []:
            if isinstance(item, dict) and item.get('item_id') not in menu:
                return 'items', 'items[].item_id phải thuộc menu đã đọc của đúng restaurant_id; gọi food_delivery.get_menu.'
    if name == 'subscriptions.cancel' and re.search('đám mây|lưu trữ', request, re.I):
        records = candidates.get('subscription_id', {}).get('records', [])
        matches = [item for item in records if item.get('cycle') == 'monthly'
                   and re.search(r'\d+\s*(GB|TB)', item.get('name', ''), re.I)
                   and item.get('status') == 'active']
        if len(matches) > 1 and not any(item.get('name', '').casefold() in request.casefold()
                                       for item in matches) and not re.search(r'\d+\s*(GB|TB)', request, re.I):
            return 'subscription_id', 'Nhiều gói lưu trữ cùng phù hợp; cần ambiguous_target và hỏi subscription_id.'
    return None, None


def read_invalidated(read_name, later):
    """The simulator does not mutate unrelated apps on a write."""
    return (later.get('kind') == 'write' and result_succeeded(later['result'])
            and later['name'].split('.')[0] == read_name.split('.')[0])
