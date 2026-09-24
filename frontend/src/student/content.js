// Student-facing copy. Country facts are condensed from backend/business/countries.yaml
// (a draft reference, re-verified by advisors each season), so every line stays general.

// Ordered north to south by the latitude of the capital.
export const COUNTRIES = [
  {
    name: '芬兰', en: 'Finland', city: '赫尔辛基', lat: 60.17,
    line: '一次联合申请最多填 6 个志愿，学费减免奖学金覆盖面很广。主申请季在 12 月到次年 1 月初。',
    schools: 'Aalto、赫尔辛基大学、坦佩雷大学',
    ask: '芬兰的英语授课计算机硕士怎么申请？学费减免奖学金好拿吗？'
  },
  {
    name: '瑞典', en: 'Sweden', city: '斯德哥尔摩', lat: 59.33,
    line: '统一平台一次最多申 4 个项目，1 月中旬截止。工作与生活的平衡很好，毕业后可以申请找工作居留。',
    schools: 'KTH、Chalmers、Lund、Uppsala',
    ask: '瑞典的英语授课计算机硕士怎么申请？什么时候截止？'
  },
  {
    name: '丹麦', en: 'Denmark', city: '哥本哈根', lat: 55.68,
    line: '各校独立申请，多数不收申请费。科技产业发达，毕业后留下工作的政策友好。',
    schools: 'DTU、哥本哈根大学、奥胡斯大学',
    ask: '丹麦的计算机硕士申请有什么要求？毕业后好留下来工作吗？'
  },
  {
    name: '德国', en: 'Germany', city: '柏林', lat: 52.52,
    line: '多数公立大学几乎不收学费，工科很强。流程最长，要先过 APS，最好提前半年开始准备。',
    schools: 'TUM、RWTH、KIT、TU Berlin',
    ask: '申请德国计算机硕士，APS 和 uni-assist 要怎么安排时间？'
  },
  {
    name: '荷兰', en: 'Netherlands', city: '阿姆斯特丹', lat: 52.37,
    line: '英语普及率高，项目偏研究型，和产业结合紧密。住房紧张，拿到录取要尽早找房。',
    schools: 'TU Delft、UvA 与 VU、TU/e、Leiden',
    ask: '荷兰计算机硕士对本科院校和均分有什么要求？'
  }
]

// Shown on the menu, in this order. Prices come live from /catalog; these are the fallback.
export const MENU = [
  { sku: 'intro_call', name: '免费初步沟通', price: 0, unit: '15 分钟', summary: '了解你的背景和目标，看看我们能不能帮上忙。' },
  { sku: 'consult_single', name: '单次选校咨询', price: 699, unit: '60 分钟', summary: '已经有初步想法，想找过来人把方向快速定下来。' },
  { sku: 'selection_full', name: '选校定位全案', price: 2999, unit: '一套', summary: '从背景评估到 8–12 个项目的选校清单和申请时间表。' },
  { sku: 'essay_pack_3', name: '文书套餐 · 3 个项目', price: 5999, unit: '一套', summary: '一篇主文书打磨透，再为 3 个目标项目分别改写。' },
  { sku: 'full_journey', name: '全程陪跑', price: 11800, unit: '一套', summary: '从选校到拿录取，一位创始人顾问全程跟进。' },
  { sku: 'full_journey_plus', name: '全程陪跑 Plus', price: 15800, unit: '一套', summary: '再加上模拟面试、签证材料和落地前的北欧生活答疑。' }
]

export const PROMISES = [
  { title: '创始人亲自做', body: '选校和文书都由我们几个在瑞典读 CS 的人完成，不外包，也不走流水线。' },
  { title: '每季只带 15 位', body: '全程陪跑每个申请季最多接 15 名学员，满了就按预约顺序排队。' },
  { title: '不代写，不保录', body: '文书由你本人执笔，我们给结构和修改意见；录取结果谁也不该打包票。' },
  { title: '人在斯德哥尔摩', body: '顾问在瑞典时区，工作日 24 小时内回复，比北京时间晚 6 到 7 小时。' }
]

export const STARTERS = [
  '瑞典的计算机硕士什么时候截止？',
  '我均分 85、雅思 7，适合申哪些学校？',
  '选校全案和全程陪跑有什么区别？',
  '我想预约一次免费沟通'
]
