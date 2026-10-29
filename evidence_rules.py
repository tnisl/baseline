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
                             or name in prop.get('description', '')
                             or name.rsplit('.', 1)[-1] == 'get_' + collection.rstrip('s')
                             or (ref == 'ride_hailing.rides' and name == 'ride_hailing.book'))):
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


def write_evidence_issue(name, arguments, spec, request, row, observations):
    """Bounded ambiguity/history checks; return issue and optional prior proof."""
    from common import folded_tokens
    text = request.casefold()
    candidates = reference_candidates(spec, row, observations)
    info = candidates.get('contact_id', {})
    if info.get('ref') == 'contacts.contacts' and arguments.get('contact_id'):
        words = folded_tokens(request)
        selected = next((item for item in info['records'] if item.get('id') == arguments['contact_id']), None)
        if selected:
            names = folded_tokens(selected.get('name', ''))
            same_name = [item for item in info['records']
                         if folded_tokens(item.get('name', '')) == names]
            if not any(folded_tokens(item.get('relation', ''))
                       and folded_tokens(item['relation']) <= words for item in same_name):
                for turn in reversed(row.get('history') or []):
                    if turn.get('role') != 'user':
                        continue
                    history_words = folded_tokens(turn.get('audio_text', ''))
                    if names and names <= history_words and any(
                            folded_tokens(item.get('relation', ''))
                            and folded_tokens(item['relation']) <= history_words for item in same_name):
                        words |= history_words
                        break
            matches = [item for item in info['records']
                       if folded_tokens(item.get('name', '')) == names]
            qualified = [item for item in matches if folded_tokens(item.get('relation', ''))
                         and folded_tokens(item['relation']) <= words]
            explicit = (str(arguments['contact_id']) in request
                        or selected.get('phone') and selected['phone'] in request)
            if len(matches) > 1 and not explicit and (len(qualified) != 1 or qualified[0]['id'] != selected['id']):
                return {'type': 'ambiguous_target', 'fields': ['contact_id'],
                        'reason': 'Nhiều liên hệ cùng tên; cần user chọn người/quan hệ hoặc số điện thoại.'}, None
    if name in ('email.create_draft', 'email.send') and len(arguments.get('to') or []) > 1:
        if re.search(r'một (?:mối|người|nơi)|chỉ (?:một|1)', text):
            return {'type': 'ambiguous_target', 'fields': ['to'],
                    'reason': 'User chỉ chọn một người nhận nhưng chưa xác định ai; không gửi cho toàn bộ ứng viên.'}, None
    # Conditional "if not done" requests must use prior successful tool evidence.
    if name == 'banking.transfer' and re.search(r'chưa|đã.*chưa', text):
        keys = ('from_account_id', 'beneficiary_id', 'amount')
        for turn in reversed(row.get('history') or []):
            for call, result in zip(turn.get('tool_calls') or [], turn.get('tool_results') or []):
                if (call.get('name') == name and result_succeeded(result)
                        and all(call.get('arguments', {}).get(key) == arguments.get(key) for key in keys)):
                    prior = {'name': name, 'kind': 'write', 'arguments': call['arguments'], 'result': result}
                    return {'type': 'already_completed', 'fields': ['beneficiary_id'],
                            'reason': 'Giao dịch này đã thành công trong history; điều kiện chưa chuyển không thỏa.'}, prior
    return None, None


def lottery_reference_issue(name, request, row, observations):
    """Do not select one of multiple trips linked to the same named person."""
    if name != 'lottery.check_ticket' or not re.search('công tác', request, re.I):
        return None
    if re.search(r'\d{4}-\d{2}-\d{2}|\b\d{1,2}/\d{1,2}', request):
        return None
    people = re.findall(r'\b[A-ZĐ][\w]+(?:\s+[A-ZĐ][\w]+)+', request)
    dates = set()
    for tool, _, result in visible_results(row, observations):
        if tool != 'calendar.list_events':
            continue
        for event in collection_records(result, 'events'):
            if any(person in event.get('title', '') for person in people) and event.get('start'):
                dates.add(event['start'][:10])
    if len(dates) > 1:
        return {'type': 'ambiguous_target', 'fields': ['date'],
                'reason': 'Có nhiều ngày công tác cùng người được tham chiếu; cần user xác định ngày xổ.'}
    return None
