import { useMemo } from 'react'

/* Priority badge color */
const priorityBadge = (p) => {
  if (p === 1) return 'bg-red-600'
  if (p === 2) return 'bg-orange-500'
  if (p === 3) return 'bg-sky-500'
  if (p === 4) return 'bg-emerald-500'
  return 'bg-gray-400'
}

/* ── Sub-components ────────────────────────────── */

function StatCards({ plans, conflicts, pushed, deadlineData }) {
  const cards = [
    {
      label: 'Total Orders',
      value: plans.length,
      icon: (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
        </svg>
      ),
      color: 'bg-slate-800 text-white',
      sub: plans.length === 1 ? 'production order' : 'production orders',
    },
    {
      label: 'On Time',
      value: `${deadlineData.onTime.length}/${plans.length}`,
      icon: (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      color: deadlineData.allMet ? 'bg-emerald-600 text-white' : 'bg-red-600 text-white',
      sub: deadlineData.allMet ? 'all deadlines met' : `${deadlineData.late.length} at risk`,
      alert: !deadlineData.allMet,
    },
    {
      label: 'Conflicts',
      value: conflicts.length,
      icon: (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
        </svg>
      ),
      color: conflicts.length > 0 ? 'bg-amber-500 text-white' : 'bg-gray-200 text-gray-500',
      sub: 'EDF vs priority swaps',
    },
    {
      label: 'Pushed',
      value: `${pushed}/${plans.length}`,
      icon: (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M7 11l5-5m0 0l5 5m-5-5v12" />
        </svg>
      ),
      color: pushed === plans.length && plans.length > 0 ? 'bg-emerald-600 text-white' : 'bg-blue-600 text-white',
      sub: pushed === plans.length && plans.length > 0 ? 'all synced to Arke' : 'synced to Arke',
    },
  ]

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {cards.map(c => (
        <div key={c.label} className={`rounded-xl shadow-sm border p-5 flex items-center gap-4 transition ${
          c.alert
            ? 'bg-red-50 border-red-200'
            : 'bg-white border-gray-200'
        }`}>
          <div className={`rounded-xl p-2.5 ${c.color} flex-shrink-0`}>
            {c.icon}
          </div>
          <div className="min-w-0">
            <div className={`text-3xl font-bold leading-tight ${c.alert ? 'text-red-800' : 'text-gray-900'}`}>{c.value}</div>
            <div className={`text-xs font-medium mt-1 ${c.alert ? 'text-red-500' : 'text-gray-400'}`}>{c.sub}</div>
          </div>
        </div>
      ))}
    </div>
  )
}

function NoticesCard({ notices }) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      <div className="px-5 py-3 border-b border-gray-100 bg-gray-50">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-800 uppercase tracking-wide">Notices</h3>
          <span className="text-xs font-medium text-gray-500 bg-gray-200 rounded-full px-2 py-0.5">{notices.length}</span>
        </div>
      </div>
      <div className="divide-y divide-gray-100">
        {notices.map((n, i) => {
          const bg = n.level === 'error' ? 'bg-red-50' : n.level === 'warning' ? 'bg-amber-50/60' : ''
          const dot = n.level === 'error' ? 'bg-red-400' : n.level === 'warning' ? 'bg-amber-400' : 'bg-slate-400'
          const txt = n.level === 'error' ? 'text-red-700' : n.level === 'warning' ? 'text-amber-800' : 'text-gray-600'
          return (
            <div key={i} className={`px-5 py-3 flex items-center gap-3 ${bg}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${dot} flex-shrink-0`}></span>
              <p className={`text-sm leading-relaxed ${txt}`}>{n.text}</p>
            </div>
          )
        })}
        {notices.length === 0 && (
          <div className="px-5 py-8 text-center text-sm text-gray-400">No notices</div>
        )}
      </div>
    </div>
  )
}

function RecentActivityCard({ recentLog, logCfg }) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      <div className="px-5 py-3 border-b border-gray-100 bg-gray-50">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-800 uppercase tracking-wide">Recent Activity</h3>
          <span className="text-xs font-medium text-gray-500 bg-gray-200 rounded-full px-2 py-0.5">{recentLog.length}</span>
        </div>
      </div>
      <div className="divide-y divide-gray-100">
        {recentLog.map((entry) => {
          const cfg = entry.category === 'summary'
            ? { dot: 'bg-teal-400' }
            : (logCfg[entry.level] || logCfg.info)
          return (
            <div key={entry.id} className="px-5 py-2.5 flex items-start gap-3 hover:bg-gray-50 transition">
              <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot} mt-2 flex-shrink-0`}></span>
              <div className="flex-1 min-w-0">
                <p className="text-xs text-gray-800 font-medium truncate">{entry.title}</p>
                {entry.detail && entry.category !== 'summary' && (
                  <p className="text-[11px] text-gray-400 truncate mt-0.5">{entry.detail}</p>
                )}
              </div>
              <span className="text-[10px] text-gray-300 flex-shrink-0 whitespace-nowrap font-mono mt-0.5">
                {entry.timestamp && new Date(entry.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
          )
        })}
        {recentLog.length === 0 && (
          <div className="px-5 py-8 text-center text-sm text-gray-400">No log entries yet</div>
        )}
      </div>
    </div>
  )
}

function DeadlineStatusCard({ deadlineData }) {
  const fmt = (d) => d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  const allItems = [...deadlineData.late, ...deadlineData.onTime]
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      <div className="px-5 py-3 border-b border-gray-100 bg-gray-50">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-800 uppercase tracking-wide">Deadline Status</h3>
          <span className={`text-xs font-medium rounded-full px-2 py-0.5 ${
            deadlineData.allMet ? 'text-emerald-700 bg-emerald-100' : 'text-red-700 bg-red-100'
          }`}>
            {deadlineData.allMet ? '\u2713 All clear' : `${deadlineData.late.length} at risk`}
          </span>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-[12px]">
          <thead>
            <tr className="border-b border-gray-100 text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
              <th className="px-5 py-2">Order</th>
              <th className="px-3 py-2">P</th>
              <th className="px-3 py-2">Ends</th>
              <th className="px-3 py-2">Deadline</th>
              <th className="px-3 py-2 text-right">Slack</th>
              <th className="px-5 py-2 text-right">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {allItems.map(p => {
              const isLate = p.slackDays < 0
              return (
                <tr key={p.order_number} className={`hover:bg-gray-50 transition ${isLate ? 'bg-red-50/50' : ''}`}>
                  <td className="px-5 py-2 font-semibold text-gray-800">{p.order_number}</td>
                  <td className="px-3 py-2">
                    <span className={`${priorityBadge(p.priority)} text-white text-[9px] font-bold px-1.5 py-0.5 rounded`}>P{p.priority}</span>
                  </td>
                  <td className="px-3 py-2 font-mono text-gray-600">{fmt(p.endsDate)}</td>
                  <td className="px-3 py-2 font-mono text-gray-600">{fmt(p.deadlineDate)}</td>
                  <td className={`px-3 py-2 text-right font-mono font-semibold ${isLate ? 'text-red-600' : 'text-emerald-600'}`}>
                    {isLate ? `${Math.abs(p.slackDays)}d late` : `+${p.slackDays}d`}
                  </td>
                  <td className="px-5 py-2 text-right">
                    {isLate ? (
                      <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-red-600 bg-red-100 rounded-full px-2 py-0.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-red-500"></span>Late
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-emerald-600 bg-emerald-100 rounded-full px-2 py-0.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>OK
                      </span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function ProductMixCard({ productBreakdown }) {
  const maxQty = Math.max(...productBreakdown.map(x => x.totalQty), 1)
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      <div className="px-5 py-3 border-b border-gray-100 bg-gray-50">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-800 uppercase tracking-wide">Product Mix</h3>
          <span className="text-xs font-medium text-gray-500 bg-gray-200 rounded-full px-2 py-0.5">{productBreakdown.length} types</span>
        </div>
      </div>
      <div className="p-5 space-y-4">
        {productBreakdown.map(pb => {
          const pct = (pb.totalQty / maxQty) * 100
          return (
            <div key={pb.product}>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-semibold text-gray-700">{pb.product}</span>
                <span className="text-[11px] text-gray-400 font-mono">
                  {pb.count} order{pb.count > 1 ? 's' : ''} &middot; {pb.totalQty} units
                </span>
              </div>
              <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-slate-600 to-slate-800 rounded-full transition-all duration-500"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          )
        })}
        {productBreakdown.length === 0 && (
          <p className="text-sm text-gray-400 text-center py-4">No products in schedule</p>
        )}
      </div>
    </div>
  )
}

function EdfReasoningCard({ plans, deadlineData }) {
  if (plans.length <= 1) return null
  const hp = plans.reduce((a, b) => (a.priority < b.priority ? a : b), plans[0])
  const hpIdx = plans.indexOf(hp)
  if (hpIdx === 0) return null
  const before = plans.slice(0, hpIdx)

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      <div className="px-5 py-3 border-b border-gray-100 bg-gray-50">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-800 uppercase tracking-wide">EDF Scheduling Rationale</h3>
          <span className="inline-flex items-center gap-1.5 text-xs font-medium text-blue-600 bg-blue-50 rounded-full px-2.5 py-0.5">
            &#x1F4CC; Insight
          </span>
        </div>
      </div>

      <div className="bg-gradient-to-r from-slate-800 to-slate-700 px-6 py-5 text-white">
        <p className="text-sm leading-relaxed">
          <span className="font-bold text-blue-300">{hp.order_number}</span>{' '}
          <span className="text-slate-300">({hp.product},</span>{' '}
          <span className={`inline-flex items-center text-[10px] font-bold px-1.5 py-0.5 rounded ${
            hp.priority <= 2 ? 'bg-red-500/30 text-red-200' : 'bg-emerald-500/30 text-emerald-200'
          }`}>P{hp.priority}</span>
          <span className="text-slate-300">)</span>{' '}
          is the highest-priority order but is scheduled at position{' '}
          <span className="font-bold text-white">#{hpIdx + 1}</span>.
        </p>
        <p className="text-sm text-slate-400 mt-2">
          {hpIdx} order{hpIdx > 1 ? 's have' : ' has'} earlier deadline{hpIdx > 1 ? 's' : ''} and {hpIdx > 1 ? 'are' : 'is'} scheduled first:
        </p>
      </div>

      <div className="px-4 py-3 bg-slate-900 border-t border-slate-700">
        <div className="flex flex-wrap gap-2">
          {before.map(b => (
            <div
              key={b.order_number}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-600 bg-slate-800 px-3 py-1.5"
            >
              <span className={`${priorityBadge(b.priority)} text-white text-[9px] font-bold px-1 py-0.5 rounded`}>P{b.priority}</span>
              <span className="text-xs font-semibold text-slate-200">{b.order_number}</span>
              <span className="text-[10px] text-slate-400 font-mono">
                {new Date(b.deadline).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="px-6 py-2.5 bg-gray-50 border-t border-gray-100">
        <p className="text-xs text-gray-500 italic">
          EDF prioritises tighter deadlines over priority level, ensuring no earlier deadline is missed.
          {deadlineData.allMet ? ' \u2705 All deadlines are met.' : ' \u26A0\uFE0F Some deadlines cannot be met.'}
        </p>
      </div>
    </div>
  )
}

/* ── Main Component ────────────────────────────── */

export default function DashboardTab({ schedule, productionLog, pushStatus }) {
  const plans = useMemo(() => schedule?.production_plans || [], [schedule])
  const conflicts = useMemo(() => schedule?.conflicts || [], [schedule])

  const deadlineData = useMemo(() => {
    const onTime = []
    const late = []
    for (const p of plans) {
      const ends = new Date(p.ends_at)
      const deadline = new Date(p.deadline)
      const item = {
        ...p,
        endsDate: ends,
        deadlineDate: deadline,
        slackDays: Math.ceil((deadline - ends) / (1000 * 60 * 60 * 24)),
      }
      if (ends > deadline) late.push(item)
      else onTime.push(item)
    }
    late.sort((a, b) => a.slackDays - b.slackDays)
    onTime.sort((a, b) => a.slackDays - b.slackDays)
    return { onTime, late, allMet: late.length === 0 }
  }, [plans])

  const notices = useMemo(() => {
    const n = []
    if (deadlineData.late.length > 0) {
      n.push({ level: 'error', icon: '\u26A0', text: `${deadlineData.late.length} order(s) will miss their deadline` })
    }
    if (deadlineData.allMet) {
      n.push({ level: 'success', icon: '\u2713', text: 'All deadlines are met under current schedule' })
    }
    if (conflicts.length > 0) {
      n.push({ level: 'warning', icon: '\u21C4', text: `${conflicts.length} EDF vs Priority conflict(s) resolved` })
    }
    const pushedCount = Object.values(pushStatus || {}).filter(r => r.status === 'pushed').length
    if (pushedCount === plans.length && plans.length > 0) {
      n.push({ level: 'success', icon: '\uD83D\uDE80', text: `All ${pushedCount} orders pushed to Arke` })
    } else if (pushedCount > 0) {
      n.push({ level: 'info', icon: '\u2191', text: `${pushedCount} / ${plans.length} orders pushed to Arke` })
    } else if (plans.length > 0) {
      n.push({ level: 'warning', icon: '\u25CB', text: 'No orders pushed to Arke yet' })
    }
    if (plans.length > 1) {
      const hp = plans.reduce((a, b) => (a.priority < b.priority ? a : b), plans[0])
      const hpIdx = plans.indexOf(hp)
      if (hpIdx > 0) {
        n.push({ level: 'info', icon: '\uD83D\uDCCC', text: `${hp.order_number} (P${hp.priority}) is scheduled #${hpIdx + 1} \u2014 EDF prefers earlier deadlines` })
      }
    }
    return n
  }, [deadlineData, conflicts, pushStatus, plans])

  const recentLog = useMemo(() => [...productionLog].slice(-8).reverse(), [productionLog])

  const productBreakdown = useMemo(() => {
    const map = {}
    for (const p of plans) {
      const key = p.product || 'Unknown'
      if (!map[key]) map[key] = { product: key, count: 0, totalQty: 0 }
      map[key].count++
      map[key].totalQty += p.quantity || 0
    }
    return Object.values(map).sort((a, b) => b.totalQty - a.totalQty)
  }, [plans])

  const pushed = Object.values(pushStatus || {}).filter(r => r.status === 'pushed').length
  const onTimePct = plans.length > 0 ? Math.round((deadlineData.onTime.length / plans.length) * 100) : 0
  const pushedPct = plans.length > 0 ? Math.round((pushed / plans.length) * 100) : 0

  const logCfg = {
    info:    { dot: 'bg-blue-400' },
    warning: { dot: 'bg-amber-400' },
    change:  { dot: 'bg-violet-400' },
    error:   { dot: 'bg-red-400' },
  }

  return (
    <div className="space-y-4">
      <StatCards plans={plans} conflicts={conflicts} pushed={pushed} deadlineData={deadlineData} />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <NoticesCard notices={notices} />
        <RecentActivityCard recentLog={recentLog} logCfg={logCfg} />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <DeadlineStatusCard deadlineData={deadlineData} />
        <ProductMixCard productBreakdown={productBreakdown} />
      </div>
      <EdfReasoningCard plans={plans} deadlineData={deadlineData} />
    </div>
  )
}
