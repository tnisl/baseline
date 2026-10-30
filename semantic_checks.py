"""Bounded semantic checks on visible evidence; no gold labels or hidden state."""
import re
from datetime import datetime, timedelta
from common import result_succeeded, normalize_vi


def visible_user_text(request, row):
    return '\n'.join([*(t.get('audio_text', '') for t in row.get('history', [])
                        if t.get('role') == 'user'), request])


def spoken_clock(text):
    # Conservative: explicit hour and minute words, no guessed AM/PM.
    numbers = {'không': 0, 'một': 1, 'hai': 2, 'ba': 3, 'bốn': 4, 'năm': 5,
               'sáu': 6, 'bảy': 7, 'tám': 8, 'chín': 9, 'mười': 10}
    pattern = r'(\d{1,2}|mười|chín|tám|bảy|sáu|năm|bốn|ba|hai|một)\s*(?:giờ|rưỡi)(?:\s*(mười lăm|hai mươi|ba mươi|bốn mươi|năm mươi|mười))?'
    values = []
    for m in re.finditer(pattern, text.casefold()):
        hour = int(m[1]) if m[1].isdigit() else numbers[m[1]]
        minute = 30 if 'rưỡi' in m[0] else {'mười lăm': 15, 'hai mươi': 20, 'ba mươi': 30,
                    'bốn mươi': 40, 'năm mươi': 50, 'mười': 10}.get(m[2], 0)
        values.append(hour * 60 + minute)
    return values


def semantic_write_error(name, args, request, row, observations):
    for field, expected in temporal_values(name, request, row, observations).items():
        if args.get(field) != expected:
            return field, f'Mốc thời gian suy từ ngày/giờ và khoảng lùi rõ ràng là {expected}.'
    if name == 'calendar.update_event':
        events = [e for o in observations if o['name'] == 'calendar.list_events'
                  and result_succeeded(o['result']) for e in o['result'].get('events', [])
                  if e.get('id') == args.get('event_id')]
        if len(events) == 1:
            merged = {**events[0], **args}
            try:
                if datetime.fromisoformat(merged['end']) <= datetime.fromisoformat(merged['start']):
                    return 'end', 'Giờ kết thúc phải sau giờ bắt đầu; cần cập nhật đủ khoảng thời gian.'
            except (KeyError, ValueError, TypeError):
                pass
    if name == 'tv.tune' and re.search(r'không phải.*kênh.*(?:ghi|thu)', request, re.I):
        excluded = {res['recording'].get('channel') for t in row.get('history', [])
                    for call, res in zip(t.get('tool_calls', []), t.get('tool_results', []))
                    if call.get('name') == 'tv.record' and result_succeeded(res) and res.get('recording')}
        if args.get('channel') in excluded:
            return 'channel', 'Kênh này bị user loại trừ vì đã hẹn ghi chương trình trước đó.'
    if name == 'learning.log_progress' and 'đến' in request:
        # Last corrected start before "đến" takes precedence.
        before, after = request.casefold().split('đến', 1)
        starts, ends = spoken_clock(before), spoken_clock(after)
        if starts and ends and 0 < ends[0] - starts[-1] <= 720:
            expected = ends[0] - starts[-1]
            if args.get('minutes') != expected:
                return 'minutes', f'Khoảng học đã nêu là {expected} phút, không cộng tiến độ cũ.'
    if name == 'bill_payment.pay' and re.search(r'chuyển.*(?:cho|tới)', request, re.I):
        return 'bill_id', 'User yêu cầu chuyển tiền cho người thụ hưởng; cần banking.transfer thay trả hóa đơn trực tiếp.'
    if name == 'learning.schedule_lesson' and re.search(r'vừa học|tranh thủ học|đã học', request, re.I):
        return 'time', 'Yêu cầu ghi nhận thời gian đã học, không tạo lịch học mới.'
    return None, None


def temporal_values(name, request, row, observations):
    """Only unique weekday/clock/route relationships; ambiguous cases stay with the model."""
    try:
        now = datetime.fromisoformat(row['context']['current_time'])
    except (KeyError, ValueError, TypeError):
        return {}
    users = [t.get('audio_text', '') for t in row.get('history', []) if t.get('role') == 'user']
    results = []
    for t in row.get('history', []):
        results.extend(zip(t.get('tool_calls', []), t.get('tool_results', [])))
    results.extend(({'name': o['name'], 'arguments': o.get('arguments', {})}, o['result']) for o in observations)
    weekdays = {'hai': 0, 'ba': 1, 'tư': 2, 'năm': 3, 'sáu': 4, 'bảy': 5}
    def date_for(text):
        days = {weekdays[m] for m in re.findall(r'thứ\s+(hai|ba|tư|năm|sáu|bảy)\b', text.casefold())}
        return now.date() + timedelta(days=(next(iter(days)) - now.weekday()) % 7) if len(days) == 1 else None
    if name == 'package_tracking.schedule_pickup' and re.search(r'trước.*(?:một|1) ngày', request, re.I):
        texts = [res.get('messages', []) for call, res in results if call.get('name') == 'sms.read_recent' and result_succeeded(res)]
        dates = {date_for(m.get('content', '')) for group in texts for m in group if re.search(r'bay', m.get('content', ''), re.I)} - {None}
        if len(dates) == 1:
            return {'date': (next(iter(dates)) - timedelta(days=1)).isoformat()}
    if name == 'video_meeting.create' and re.search(r'đúng giờ đó|giờ họp', request, re.I):
        meetings = [t for t in users if re.search(r'\bhọp\b', t, re.I)]
        if len(meetings) == 1:
            day, clocks = date_for(meetings[0]), spoken_clock(meetings[0])
            if day and len(clocks) == 1:
                value = datetime.combine(day, now.timetz()).replace(hour=clocks[0] // 60, minute=clocks[0] % 60, second=0, microsecond=0)
                return {'start': value.isoformat()}
    if name == 'reminders.create' and re.search(r'nhắc.*rời trạm', request, re.I):
        appointments = [t for t in users if 'hẹn' in t.casefold() and date_for(t)]
        routes = [res['route'] for call, res in results if call.get('name') == 'maps.get_route'
                  and result_succeeded(res) and isinstance(res.get('route', {}).get('duration_min'), (int, float))]
        if len(appointments) == 1 and len(routes) == 1:
            clocks = spoken_clock(appointments[0])
            if len(clocks) == 1:
                day = date_for(appointments[0])
                arrival = datetime.combine(day, now.timetz()).replace(hour=clocks[0] // 60, minute=clocks[0] % 60, second=0, microsecond=0)
                return {'time': (arrival - timedelta(minutes=routes[0]['duration_min'])).isoformat()}
    return {}


def goal_result_error(task, observation):
    """Check only unambiguous action language; generic goals remain unclassified."""
    text = task.casefold()
    rules = [
        (r'chuyển tiền', {'banking.transfer'}),
        (r'(?:bật|chỉnh|mở).*?(?:điều hoà|điều hòa|máy lạnh)', {'smart_home.climate_set'}),
        (r'(?:gửi|nhắn).*?sms', {'sms.send'}),
        (r'(?:gửi|nhắn).*?zalo', {'chat_ott.send_message'}),
        (r'(?:trả lời|rep).*?(?:email|mail)', {'email.reply'}),
    ]
    # A lookup goal mentioning an action is not itself that action.
    if observation.get('kind') != 'write':
        return None
    for pattern, tools in rules:
        if re.search(pattern, text) and observation['name'] not in tools:
            return 'Bằng chứng tool không thực hiện hành động của goal.'
    if observation['name'] == 'bill_payment.pay':
        kinds = [kind for phrase, kind in [('tiền nước', 'water'), ('tiền mạng', 'internet'),
                                           ('truyền hình', 'tv')] if phrase in text]
        bill = observation['result'].get('bill', {})
        if len(kinds) == 1 and bill.get('type') and bill['type'] != kinds[0]:
            return 'Hóa đơn đã trả khác loại hóa đơn trong goal.'
    return None
