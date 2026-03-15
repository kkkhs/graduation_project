const SHANGHAI_FORMATTER = new Intl.DateTimeFormat('zh-CN', {
  timeZone: 'Asia/Shanghai',
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false
})

export function formatShanghaiTime(value: string | null | undefined): string {
  if (!value) {
    return '-'
  }

  // Backend stores naive UTC timestamps via datetime.utcnow(), so append Z to enforce UTC parse.
  const normalized = /[zZ]|[+-]\d{2}:\d{2}$/.test(value) ? value : `${value}Z`
  const date = new Date(normalized)
  if (Number.isNaN(date.getTime())) {
    return '-'
  }

  return SHANGHAI_FORMATTER.format(date)
}
