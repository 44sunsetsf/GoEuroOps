// 后端枚举值 → 界面文案

export const AGENT_LABELS = {
  general: '前台接待',
  consulting: '留学咨询',
  billing: '费用售后',
  escalation: '转顾问'
}

export const INTENT_LABELS = {
  greeting: '问候',
  query: '一般询问',
  request: '请求操作',
  complaint: '投诉',
  feedback: '反馈',
  escalation: '升级',
  study_consult: '留学咨询',
  application_process: '申请流程',
  service_inquiry: '服务与价格',
  booking: '预约',
  service_progress: '服务进度',
  billing: '费用',
  refund: '退款',
  invoice: '发票',
  payment_issue: '付款异常',
  account: '联系方式',
  data_privacy: '隐私',
  human_handoff: '转人工',
  other: '其他'
}

export const RAG_MODE_LABELS = {
  prefetch: '知识库 · 预取',
  on_demand: '知识库 · 按需',
  off: '不检索',
  skipped: '未检索'
}

export const LEAD_STATUS = [
  { value: 'new', label: '待联系' },
  { value: 'contacted', label: '已联系' },
  { value: 'converted', label: '已成交' },
  { value: 'closed', label: '已关闭' }
]

export const LEAD_TYPE_LABELS = { lead: '咨询线索', handoff: '转顾问' }
export const CHANNEL_LABELS = { wechat: '微信', email: '邮箱', phone: '手机' }
export const STAGE_LABELS = {
  exploring: '了解中',
  preparing: '准备中',
  applying: '申请中',
  admitted: '已录取'
}

export const DOMAIN_OPTIONS = [
  { value: '', label: '全部 / 共享' },
  { value: 'consulting', label: '留学咨询' },
  { value: 'billing', label: '费用售后' },
  { value: 'general', label: '前台接待' },
  { value: 'shared', label: '共享' }
]

export const CATEGORY_ORDER = ['consult', 'selection', 'essay', 'package', 'addon']
export const CATEGORY_LABELS = {
  consult: '咨询',
  selection: '选校',
  essay: '文书单项',
  package: '套餐',
  addon: '增值服务'
}

export function label(map, value) {
  return map[value] || value || '-'
}

export function formatTime(iso) {
  if (!iso) return '-'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}

export function money(value) {
  const number = Number(value || 0)
  return number === 0 ? '免费' : `¥${number.toLocaleString('zh-CN')}`
}
