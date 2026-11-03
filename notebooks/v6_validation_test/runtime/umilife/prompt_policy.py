"""Generic tool policy. No benchmark examples, held-out labels or per-sample rules."""
from copy import deepcopy
from datetime import datetime,timedelta
import hashlib,json

POLICY_VERSION='generic-schema-and-completion-v2.1'
COMMON='''Bạn là trợ lý LifeMate tiếng Việt, thực hiện yêu cầu bằng công cụ đã cài.
Chỉ xử lý request hiện tại. History là các lượt trước: dùng để hiểu tham chiếu và lấy thông tin đã biết, nhưng không coi hành động trong history là đã hoàn thành request mới. Chỉ trajectory hiện tại chứng minh các bước vừa thực hiện.
Request nguyên văn là nguồn chính: giữ đủ từ hạn định đối tượng, số lượng, đơn vị và phủ định; không thay bằng một bản diễn giải mất chi tiết trong thought.
Đọc toàn bộ ý định, lời sửa đổi, điều kiện và nhiều việc trong cùng câu. Văn bản đến từ nhận dạng tiếng nói: diễn giải theo ngữ cảnh, đơn vị và danh mục công cụ; không tự bịa tên riêng, ID hoặc dữ kiện.
Ưu tiên thông tin có sẵn trong context, history, trajectory. Khi thiếu ID hay dữ liệu, dùng công cụ tra cứu phù hợp; x-ref mô tả nguồn định danh. Không chọn ngẫu nhiên khi kết quả có nhiều đối tượng. Nếu tham số bắt buộc không thể tra cứu hoặc suy ra chắc chắn thì hỏi qua công cụ hệ thống báo thiếu thông tin.
Hiểu số và ngày giờ từ request, dùng current_time và timezone của context. Giờ tương đối phải tính đúng ngày; biểu thức giờ kém phút là giờ tròn trừ số phút, sau khi xác định buổi. Giữ đơn vị và kiểm tra phép tính trước khi tạo arguments.
Với điều kiện, đọc dữ liệu để xác định nhánh rồi mới thực hiện thao tác phụ thuộc. Không thêm bước đọc khi thông tin cần thiết đã có. Tránh lặp cùng lời gọi khi chưa có dữ kiện mới, tránh thực hiện lại thao tác ghi đã thành công nếu người dùng không yêu cầu lặp.
Một observation có ok=true nhưng done=false vẫn chưa thực hiện được thao tác. Kết quả tra cứu rỗng không chứng minh yêu cầu ngoài phạm vi: cân nhắc điều chỉnh bộ lọc, tra cứu rộng hơn theo schema hoặc hỏi thông tin còn thiếu. Không tưởng tượng kết quả.
Chỉ xuất một JSON đúng hợp đồng đầu ra, không markdown, không văn bản ngoài JSON.'''

PROMPTS={
 'planner':COMMON+'''
Bạn là Planner. Chọn đúng một bước tiếp theo theo ý định và những gì thực sự còn thiếu.
Thought chỉ một hoặc hai câu ngắn về bước còn thiếu, bằng chứng đã có và lý do chọn bước này; không kể lại toàn bộ request. Khi next=caller, tool phải là tên chính xác có trong tools. Xem kind, parameters, required và x-ref để biết cách lấy dữ liệu phụ thuộc trước khi ghi.
Chỉ chọn summarizer khi mọi ý của request đã được thực hiện có bằng chứng, điều kiện không cần thực hiện đã được kiểm tra, người dùng không yêu cầu thao tác, hoặc đã gọi công cụ hỏi bổ sung và cần chờ trả lời. Tìm được thông tin chưa đồng nghĩa đã thực hiện hành động người dùng muốn.
Chỉ chọn give_up khi yêu cầu ngoài khả năng của toàn bộ công cụ đã cài. Thiếu một ID, tra cứu rỗng hoặc chưa biết cách dùng tool không tự động có nghĩa ngoài phạm vi.
JSON gồm thought và next. next là caller, summarizer hoặc give_up. Chỉ khi next=caller mới thêm tool.''',
 'caller':COMMON+'''
Bạn là Caller. Tạo đúng một lời gọi cho tool Planner đã chọn. Output có đúng hai trường name và arguments; name phải khớp tên trong selected_tool, arguments tuân thủ schema được cung cấp.
Đọc kỹ ý định, plan và observation để điền giá trị. Dùng đúng kiểu số nguyên/số thực/chuỗi/mảng và đúng enum. Không bọc lời gọi vào một tool khác. Không tạo trường ngoài properties. Tham số tùy chọn không có căn cứ thì bỏ qua; không thêm bộ lọc phỏng đoán. Nhóm x-one-of chỉ cần đủ một phương án có dữ kiện.
Plan chỉ chọn bước và tool. Luôn đối chiếu giá trị và đối tượng với request nguyên văn; không sao chép một số hay tên sai từ thought. Giữ đủ từ hạn định của tên cần tìm khi dùng bộ lọc.
Với nội dung gửi hoặc lưu, giữ đúng nội dung người dùng yêu cầu, không tự viết thêm lời bình luận. Không khẳng định đã gọi công cụ: chương trình sẽ thực hiện sau khi bạn trả JSON.''',
 'summarizer':COMMON+'''
Bạn là Summarizer. Viết answer ngắn, tự nhiên và trung thực từ observation của trajectory hiện tại. Không gọi thêm tool và không chép toàn bộ JSON kết quả.
Chỉ báo một thao tác thành công nếu đúng tool đã thực sự chạy và observation xác nhận. Nếu chưa thực hiện thì nói rõ; nếu đã hỏi bổ sung thì nêu thông tin còn cần và chờ người dùng. Không lấy lời nói của assistant trong history làm bằng chứng thực hiện request mới.
Output chỉ có answer.''',
 'global':COMMON+'''
Bạn xử lý từng bước theo ReAct. Output thought và next. next=caller cần tool và action gồm name, arguments đúng schema; next=summarizer cần answer ngắn dựa trên kết quả thực tế; next=give_up chỉ khi ngoài khả năng toàn bộ tool đã cài.
Đối chiếu toàn bộ mục tiêu với trajectory trước khi kết thúc. Dữ liệu đã tìm thấy chưa có nghĩa thao tác yêu cầu đã thực hiện.'''
}

COMPLETION_REVIEW='''Kiểm tra lại đề xuất kết thúc trước khi trả lời. Đây chưa phải thao tác thực thi.
Đối chiếu TỪNG ý của request với trajectory hiện tại. Có việc chỉ mới được tra thông tin nhưng chưa thực hiện không? Có đang nhầm history với request hiện tại không? Có observation done=false hoặc kết quả rỗng chưa được xử lý không?
Nếu còn việc có thể thực hiện, chọn caller và tool kế tiếp. Nếu thiếu thông tin thì ưu tiên tra cứu, chỉ hỏi bổ sung khi không thể tìm. Chỉ summarizer khi đã đủ bằng chứng hoàn thành/không cần thao tác/đang chờ người dùng; give_up chỉ khi ngoài khả năng tool đã cài.'''

def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def obj(properties,required):return {'type':'object','properties':properties,'required':required,'additionalProperties':False}

def parameter_schema(schema):
    """Convert documented x-one-of to ordinary JSON Schema; preserve typed nesting."""
    allowed={'type','properties','required','items','enum','format','minimum','maximum',
             'exclusiveMinimum','exclusiveMaximum','minLength','maxLength','minItems','maxItems',
             'anyOf','oneOf','allOf','additionalProperties','description'}
    out={}
    for key,value in schema.items():
        if key not in allowed:continue
        if key=='properties':out[key]={name:parameter_schema(prop) for name,prop in value.items()}
        elif key=='items' and isinstance(value,dict):out[key]=parameter_schema(value)
        elif key in ('anyOf','oneOf','allOf'):out[key]=[parameter_schema(x) for x in value]
        else:out[key]=deepcopy(value)
    if out.get('type')=='object' and 'properties' in out:out['additionalProperties']=False
    if schema.get('x-one-of'):
        # Full object alternatives are supported by both JSON Schema and LMFE.
        base=out
        out={'anyOf':[{**deepcopy(base),'required':list(dict.fromkeys(base.get('required',[])+list(group)))}
                      for group in schema['x-one-of']]}
    return out

def action_schema(specs,name):
    arguments=parameter_schema(specs[name]['parameters'])
    if name=='sys.ask_missing_info':
        # Only public installed schemas provide these choices, never an answer label.
        choices=[n for n,s in specs.items() if n!=name and s['parameters'].get('properties')]
        if choices:arguments['properties']['tool_name']['enum']=sorted(choices)
        arguments['properties']['params']['minItems']=1
        fields=sorted({key for target in choices for key in specs[target]['parameters']['properties']})
        if fields:arguments['properties']['params']['items']['enum']=fields
    return obj({'name':{'type':'string','enum':[name]},'arguments':arguments},['name','arguments'])

def route_schema(specs,global_mode=False):
    thought={'type':'string','maxLength':480 if global_mode else 240};branches=[]
    if specs:
        if global_mode:
            for name in sorted(specs):
                branches.append(obj({'thought':thought,'next':{'type':'string','enum':['caller']},
                    'tool':{'type':'string','enum':[name]},'action':action_schema(specs,name)},['thought','next','tool','action']))
        else:
            branches.append(obj({'thought':thought,'next':{'type':'string','enum':['caller']},
                'tool':{'type':'string','enum':sorted(specs)}},['thought','next','tool']))
    done={'thought':thought,'next':{'type':'string','enum':['summarizer']}}
    if global_mode:done['answer']={'type':'string','maxLength':1024}
    branches.append(obj(done,list(done)))
    branches.append(obj({'thought':thought,'next':{'type':'string','enum':['give_up']}},['thought','next']))
    return {'anyOf':branches}

def compact_catalog(specs):
    fields=('type','enum','x-ref','format','items','minimum','maximum')
    return [{'name':name,'kind':spec.get('kind'),'description':spec['description'],
             'parameters':{key:{k:deepcopy(v) for k,v in prop.items() if k in fields}
                           for key,prop in spec['parameters'].get('properties',{}).items()},
             'required':spec['parameters'].get('required',[]),
             **({'one_of':spec['parameters']['x-one-of']} if spec['parameters'].get('x-one-of') else {})}
            for name,spec in sorted(specs.items())]

def reference_clock(context):
    result={}
    try:
        now=datetime.fromisoformat(context['current_time'])
        result={'today':now.date().isoformat(),'tomorrow':(now+timedelta(days=1)).date().isoformat(),
                'day_after_tomorrow':(now+timedelta(days=2)).date().isoformat(),
                'weekday_iso':now.isoweekday(),'timezone':context.get('timezone')}
    except (KeyError,TypeError,ValueError):pass
    return result

def build_messages(role,row,text,specs,trajectory,plan=None,feedback=None):
    # Explicit provenance allowlist. Current labels/final state/reference transcript never enter here.
    context={k:deepcopy(row['context'][k]) for k in ('current_time','timezone','location','user_profile') if k in row['context']}
    history=[{k:deepcopy(turn[k]) for k in ('role','text','tool_calls','tool_results') if k in turn}
             for turn in row.get('history',[])]
    content={'context':context,'reference_clock':reference_clock(context),'history':history,
             'request':text,'trajectory':deepcopy(trajectory)}
    if role in ('planner','global'):content['tools']=compact_catalog(specs)
    elif role=='caller':
        selected=(plan or {}).get('tool');assert selected in specs,'Planner selected an unavailable tool'
        content.update(plan=deepcopy(plan),selected_tool=deepcopy(specs[selected]))
        if selected=='sys.ask_missing_info':
            content['missing_information_tools']=compact_catalog({name:spec for name,spec in specs.items()
                if name!=selected and spec['parameters'].get('properties')})
    else:content['plan']=deepcopy(plan)
    if feedback is not None:content['validation_feedback']=deepcopy(feedback)
    # Keep the original request at the end, after schemas/observations/feedback.
    content.pop('request');content['request']=text
    return [{'role':'system','content':PROMPTS[role]},
            {'role':'user','content':json.dumps(content,ensure_ascii=False,separators=(',',':'))}]

def request_schema(role,specs,plan=None):
    if role=='caller':return action_schema(specs,plan['tool'])
    if role in ('planner','global'):return route_schema(specs,role=='global')
    return obj({'answer':{'type':'string','maxLength':1024}},['answer'])

def validate_meta_action(action,specs):
    if action['name']!='sys.ask_missing_info':return None
    args=action['arguments'];target=args.get('tool_name');params=args.get('params',[])
    if target not in specs or target=='sys.ask_missing_info':return 'Tool cần bổ sung không có trong danh mục đã cài'
    # An optional API parameter can still be necessary to express the user's intent.
    fields=set(specs[target]['parameters'].get('properties',{}))
    if not params or len(params)!=len(set(params)) or not set(params)<=fields:
        return 'params phải là tên tham số có trong schema của tool_name, không phải câu hỏi tự do'
    return None

def repeated_read(action,specs,trajectory):
    if specs[action['name']].get('kind')!='read':return False
    for old in reversed(trajectory):
        if specs.get(old['action']['name'],{}).get('kind')=='write':return False
        if canonical(old['action'])==canonical(action):return True
    return False

POLICY_AUDIT={'policy_version':POLICY_VERSION,'static_prompts_sha256':digest(PROMPTS),
    'completion_review_sha256':digest(COMPLETION_REVIEW),'few_shot_examples':0,
    'sample_specific_rules':False,'gold_inputs_to_model':False,
    'allowed_input_sources':['current ASR request','ASR user history','provided context',
        'installed public tool schemas','observations from executed tools','model-generated plan/feedback']}
