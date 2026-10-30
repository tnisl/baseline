"""Bounded request checks from visible inputs; no labels or hidden state.

This is a conservative contract for explicit supported commands, not a general
Vietnamese semantic parser. Unknown conditional branches remain advisory.
"""
import re

SUPPORTED_EFFECTS = frozenset({"smart_home.light_set", "smart_home.climate_set", "bill_payment.pay"})
from common import result_succeeded


def requested_light_power(request):
    text = re.split(r'còn giờ|còn bây giờ', request.casefold())[-1]
    actions = re.findall(r'(?<!tự )\b(bật|mở|tắt)\s+(?:(?:giúp|giùm|hộ|cho|tôi|mình|anh|chị|em|cô|tớ|tui|cái|các|mấy)\s+){0,5}đèn', text)
    values = {'off' if action == 'tắt' else 'on' for action in actions}
    return next(iter(values)) if len(values) == 1 else None


def intent_fields(name, request, spec):
    text = request.casefold()
    fields = set()
    if name == 'smart_home.light_set':
        if requested_light_power(request) is not None:
            fields.add('power')
        if re.search(r'phần trăm|%', text):
            fields.add('brightness')
    if name == 'smart_home.climate_set' and re.search(r'\d+\s*độ', text):
        fields.add('temperature')
    if name == 'medication.log_taken' and re.search(r'\b(?:giờ|lúc|hồi)\b', text):
        fields.add('time')
    if name == 'calendar.update_event' and re.search(r'sửa giờ|sửa.*giờ|dời|đổi.*giờ', text):
        fields.update(('start', 'end'))
    if name == 'notes.create' and re.search(r'(?:đặt tên|tên là|ghi chú tên)', text):
        fields.add('title')
    return sorted(fields & set(spec.get('parameters', {}).get('properties') or {}))


def wallet_condition(request, observations):
    """Return enough/insufficient only for a unique explicitly named bill type."""
    text = request.casefold()
    if not ('ví' in text and 'ngân hàng' in text):
        return None
    kinds = [kind for kind, phrase in [('tv', 'truyền hình'), ('internet', 'tiền mạng'),
                                      ('water', 'tiền nước')] if phrase in text]
    if len(kinds) != 1:
        return None
    bills = {}
    balance = None
    for item in observations:
        if not result_succeeded(item['result']):
            continue
        if item['name'] == 'bill_payment.list_bills':
            for bill in item['result'].get('bills', []):
                if bill.get('type') == kinds[0]:
                    bills[bill['id']] = bill
        if item['name'] == 'ewallet.get_balance':
            balance = item['result'].get('balance')
    if len(bills) == 1 and isinstance(balance, (int, float)):
        amount = next(iter(bills.values())).get('amount')
        if isinstance(amount, (int, float)):
            return balance >= amount
    return None


def request_requirements(request, specs, observations, row=None):
    text = request.casefold()
    rules = [
        ('bill_payment.pay', r'(?:trả|đóng)\b.{0,40}(?:hoá đơn|hóa đơn|tiền (?:mạng|nước|truyền hình))'),
        ('banking.transfer', r'chuyển.*(?:triệu|nghìn|ngàn).*tài khoản'),
        ('sms.send', r'(?:nhắn|gửi)\s+(?:tin\s+)?sms'),
        ('email.create_draft', r'(?:soạn|tạo).*?(?:nháp|thư nháp)'),
        ('email.reply', r'rep\s*mail|trả lời (?:email|mail)'),
        ('reminders.create', r'\bnhắc\s+(?:tôi|mình|anh|chị|cô|tui|tớ|em|con)\b'),
        ('notes.create', r'(?:tạo|lưu|ghi|chép).*ghi chú|ghi chú\s+(?:lại|giùm|giúp)'),
        ('home_automation.create_rule', r'tự\s+(?:bật|tắt).*(?:hằng|hàng|mỗi)\s+(?:ngày|tối|sáng)'),
        ('alarm.update', r'mưa.*(?:dời|đổi).*báo thức'),
        ('smart_home.climate_set', r'(?:bật|mở|chỉnh).*?(?:điều hoà|điều hòa|máy lạnh)'),
        ('smart_home.light_set', r'(?:bật|mở|tắt|chỉnh).*?đèn'),
        ('chat_ott.send_message', r'(?:nhắn|gửi).*?(?:zalo)|(?:zalo)\s+cho'),
        ('home_automation.create_rule', r'(?:hằng|hàng|mỗi)\s+(?:ngày|tối|sáng).*?tự\s+(?:bật|tắt)'),
        ('household_chores.assign', r'(?:việc đi chợ|giặt|phơi).*giao|giao.*(?:việc|giặt|phơi|đi chợ)'),
    ]
    conditional = bool(re.search(r'\bnếu\b|\bthì\b|không đủ|còn thiếu|chưa tới', text))
    wallet = wallet_condition(request, observations)
    output = []
    for tool, pattern in rules:
        match = re.search(pattern, text)
        if not match or tool not in specs:
            continue
        if tool == 'smart_home.light_set' and re.search(r'tự\s+(?:bật|tắt)|hẹn giờ', text) and not re.search(r'còn (?:bây )?giờ|ngay bây giờ', text):
            continue
        if tool == 'bill_payment.pay':
            sms = re.search(r'(?:nhắn|gửi)\s+(?:tin\s+)?sms', text)
            if sms and match.start() > sms.start():
                continue
        status = 'unknown' if conditional else 'active'
        # The explicitly supported wallet branches are evaluated from results.
        if tool == 'bill_payment.pay' and 'ví' in text and 'ngân hàng' in text:
            status = 'active'
        elif text.startswith('nếu ví đủ') and tool in ('sms.send', 'reminders.create'):
            before_else = re.split(r'còn (?:ví )?(?:thiếu|ví thiếu)|còn thiếu', text)[0]
            in_first = re.search(pattern, before_else) is not None
            status = 'unknown' if wallet is None else ('active' if wallet == in_first else 'inactive')
        elif tool == 'sms.send' and 'ví' in text and 'ngân hàng' in text and re.search(r'(?:xong|sau đó|rồi)\s+(?:rồi\s+)?nhắn\s+sms', text):
            status = 'active'
        price = gold_condition(request, observations)
        if price is not None and tool in ('reminders.create', 'banking.transfer'):
            status = ('active' if price == (tool == 'reminders.create') else 'inactive')
        if tool in ('smart_home.climate_set', 'smart_home.light_set') and not re.search(r'\bnếu\b|còn đủ|không đủ', text):
            status = 'active'
        # Only resolve a weather branch when tomorrow's forecast is visible.
        if tool == 'alarm.update':
            forecasts = [f for o in observations if o['name'] == 'weather.get_forecast'
                         and result_succeeded(o['result']) for f in o['result'].get('forecast', [])]
            if len(forecasts) == 1 and isinstance(forecasts[0].get('rain'), bool):
                status = 'active' if forecasts[0]['rain'] else 'inactive'
        requirement = {'tool': tool, 'status': status, 'evidence': match.group(0)}
        if tool == 'smart_home.climate_set':
            temperature = re.search(r'(\d+)\s*độ', text)
            if temperature:
                requirement['arguments'] = {'temperature': int(temperature[1])}
        output.append(requirement)
    targets = eligible_bill_targets(request, row, observations)
    if targets is not None and 'bill_payment.pay' in specs:
        output = [item for item in output if item['tool'] != 'bill_payment.pay']
        output.extend({'tool': 'bill_payment.pay', 'status': 'active',
                       'arguments': {'bill_id': bill_id}, 'evidence': 'Hóa đơn thỏa hạn và ngưỡng user yêu cầu'}
                      for bill_id in targets)
    # Unspecified communication channel: require a message without guessing the channel.
    if (re.search(r'\bnhắn\b', text) and not re.search(r'sms|zalo|email|mail', text)
            and not conditional):
        alternatives = [name for name in ('sms.send', 'chat_ott.send_message') if name in specs]
        if alternatives:
            output.append({'tool': alternatives[0], 'alternatives': alternatives,
                           'status': 'active', 'evidence': 'Yêu cầu nhắn thông báo'})
    return output


def request_done_error(request, specs, observations, row=None):
    pending = []
    for requirement in request_requirements(request, specs, observations, row):
        if requirement['status'] == 'active' and not any(
                item['name'] in requirement.get('alternatives', [requirement['tool']]) and result_succeeded(item['result'])
                and item.get('effect_verified') is not False
                and all(item.get('arguments', {}).get(k) == v for k, v in requirement.get('arguments', {}).items())
                for item in observations):
            pending.append(requirement['tool'] + (':' + str(requirement['arguments']['bill_id'])
                           if 'bill_id' in requirement.get('arguments', {}) else ''))
    return ('Request còn kết quả chưa có bằng chứng: ' + ', '.join(pending)) if pending else None


def intent_value_error(name, args, request):
    text = request.casefold()
    if name == 'smart_home.light_set':
        power = requested_light_power(request)
        if power is not None and args.get('power') != power:
            return 'power', 'power chưa đáp ứng thao tác bật/tắt đèn user yêu cầu.'
    return None, None


def branch_write_error(name, request, observations):
    requirements = request_requirements(request, {name: {}}, observations)
    for requirement in requirements:
        if requirement['status'] == 'inactive':
            return 'Nhánh điều kiện của hành động này không áp dụng theo dữ kiện đã đọc.'
        if requirement['status'] == 'unknown' and (request.casefold().startswith('nếu ví đủ') or gold_threshold(request) is not None):
            return 'Chưa đủ dữ kiện xác định nhánh điều kiện; đọc số dư/hóa đơn hoặc giá trước write.'
    if name == 'sms.send' and re.search(r'đã đóng|đã thanh toán', request, re.I):
        payment_requested = any(item['tool'] == 'bill_payment.pay' for item in
                                request_requirements(request, {'bill_payment.pay': {}}, observations))
        if payment_requested and not any(item['name'] == 'bill_payment.pay'
                                         and result_succeeded(item['result']) for item in observations):
            return 'Chưa có kết quả thanh toán thành công để gửi thông báo đã đóng.'
    return None


def effect_error(name, args, result):
    """Check supported effects, retaining unknown effects for unsupported tools."""
    if not result_succeeded(result):
        return None  # Existing failure handling retains the tool result.
    if name in ('smart_home.light_set', 'smart_home.climate_set'):
        state = (result.get('item') or {}).get('state') or {}
        for field in ('power', 'brightness', 'mode', 'temperature'):
            if field in args and state.get(field) != args[field]:
                return f'Result chưa xác nhận {field}={args[field]}.'
    if name == 'bill_payment.pay':
        bill = result.get('bill') or {}
        if bill.get('id') != args.get('bill_id') or bill.get('paid') is not True:
            return 'Result chưa xác nhận đúng hóa đơn đã thanh toán.'
    return None


def gold_threshold(request):
    text = request.casefold()
    match = re.search(r'bán ra.{0,30}trên\s+(\d+(?:[.,]\d+)?)\s*triệu', text)
    if match and re.search('chưa tới|chưa đến', text):
        return float(match.group(1).replace(',', '.')) * 1000000
    return None


def gold_condition(request, observations):
    threshold = gold_threshold(request)
    if threshold is None:
        return None
    kind = 'sjc' if 'sjc' in request.casefold() else ('ring' if 'vàng nhẫn' in request.casefold() else None)
    if kind is None:
        return None
    for item in reversed(observations):
        if item['name'] == 'gold_price.get_price' and result_succeeded(item['result']):
            prices = [price for price in item['result'].get('prices', []) if price.get('type') == kind]
            if len(prices) == 1 and isinstance(prices[0].get('sell'), (int, float)):
                return prices[0]['sell'] > threshold
    return None


def eligible_bill_targets(request, row, observations):
    """Only parse an explicit all-bills/date-window/upper-limit request."""
    from datetime import datetime, timedelta
    text = request.casefold()
    if not row or not re.search('tất cả|hoá đơn nào|hóa đơn nào', text):
        return None
    days = re.search(r'(\d+|năm)\s*ngày tới', text)
    limit = re.search(r'(?:hơn|vượt)\s*(\d+)\s*(nghìn|ngàn|triệu)', text)
    if not days or not limit:
        return None
    try:
        start = datetime.fromisoformat(row['context']['current_time']).date()
        end = start + timedelta(days=5 if days[1] == 'năm' else int(days[1]))
    except (KeyError, ValueError):
        return None
    maximum = int(limit[1]) * (1000000 if limit[2] == 'triệu' else 1000)
    # Preserve initial unpaid targets even after a later read reports paid=True.
    bills = {}
    seen = False
    for item in observations:
        if item['name'] == 'bill_payment.list_bills' and result_succeeded(item['result']):
            seen = True
            for bill in item['result'].get('bills', []):
                bills.setdefault(bill.get('id'), bill)
    if not seen:
        return None
    targets = []
    for bill_id, bill in bills.items():
        try:
            due = datetime.fromisoformat(bill['due_date']).date()
            if bill_id and not bill.get('paid') and start <= due <= end and bill['amount'] <= maximum:
                targets.append(bill_id)
        except (ValueError, KeyError, TypeError):
            return None
    return targets
