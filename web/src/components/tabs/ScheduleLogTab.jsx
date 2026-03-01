import { useState } from 'react'
import { formatDate } from './tabHelpers'

/* ───────────────────────────────────────────────
   Priority badge color (kept vivid)
   ─────────────────────────────────────────────── */
const priorityBadge = (p) => {
  if (p === 1) return 'bg-red-600'
  if (p === 2) return 'bg-orange-500'
  if (p === 3) return 'bg-sky-500'
  if (p === 4) return 'bg-emerald-500'
  return 'bg-gray-400'
}

/* ───────────────────────────────────────────────
   Mini order block — neutral card, colored badge
   ─────────────────────────────────────────────── */
function OrderBlock({ order, plan }) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg px-3 py-2.5 flex-1 min-w-0 shadow-[0_1px_2px_rgba(0,0,0,0.04)]">
      <div className="flex items-center gap-2 mb-1.5">
        <span className={`${priorityBadge(order.priority)} text-white text-[10px] font-bold px-1.5 py-0.5 rounded`}>P{order.priority}</span>
        <span className="font-semibold text-sm text-gray-800">{order.order}</span>
      </div>
      {plan && (
        <div className="space-y-0.5">
          <p className="text-[11px] text-gray-500">{plan.customer}</p>
          <p className="text-[11px] text-gray-500">{plan.quantity}× {plan.product}</p>
        </div>
      )}
      <div className="mt-2 pt-1.5 border-t border-gray-100 flex items-center justify-between">
        <span className="text-[10px] font-medium text-gray-400">Deadline</span>
        <span className="text-[11px] font-bold text-gray-700">{order.deadline}</span>
      </div>
    </div>
  )
}

/* ───────────────────────────────────────────────
   Collapsible conflict visual — two queue lanes
   ─────────────────────────────────────────────── */
function ConflictVisual({ conflict, plans }) {
  const [open, setOpen] = useState(false)

  if (!conflict) return null

  const priorityPlan = plans?.find(p => p.order_number === conflict.priority_first.order)
  const edfPlan      = plans?.find(p => p.order_number === conflict.edf_first.order)

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 text-[11px] font-medium text-amber-600 hover:text-amber-800 transition group"
      >
        <svg
          className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-90' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
        </svg>
        {open ? 'Hide' : 'Show'} scheduling impact
      </button>

      {open && (
        <div className="mt-3 rounded-lg border border-gray-200 bg-gray-50/50 overflow-hidden">
          {/* Priority Order (what would happen) */}
          <div className="px-4 py-3">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-[10px] font-bold uppercase tracking-wider text-gray-400">By Priority</span>
              <span className="text-[10px] text-gray-300 italic">would have been</span>
            </div>
            {/* Single spanning timeline */}
            <div className="relative mb-2">
              <div className="flex items-center">
                <span className="w-1.5 h-1.5 rounded-full bg-gray-300 flex-shrink-0 z-10"></span>
                {priorityPlan && <span className="ml-1.5 text-[9px] font-mono text-gray-400">{formatDate(priorityPlan.starts_at)}</span>}
                <div className="flex-1 mx-1.5 h-px bg-gray-200"></div>
                {priorityPlan && <span className="text-[9px] font-mono text-gray-400">{formatDate(priorityPlan.ends_at)}</span>}
                <span className="w-1.5 h-1.5 rounded-full bg-gray-300 mx-1.5 flex-shrink-0 z-10"></span>
                <div className="flex-1 mx-0 h-px bg-gray-200"></div>
                {edfPlan && <span className="text-[9px] font-mono text-gray-400">{formatDate(edfPlan.ends_at)}</span>}
                <span className="w-1.5 h-1.5 rounded-full bg-gray-400 ml-1.5 flex-shrink-0 z-10"></span>
              </div>
            </div>
            <div className="relative py-4">
              {/* Background vertical lines */}
              <div className="absolute flex justify-between" style={{ top: 0, bottom: 0, left: '-6px', right: '-6px' }} aria-hidden="true">
                {Array.from({ length: 32 }).map((_, i) => (
                  <span key={i} className="w-px h-full"
                    style={ i > 0 && i < 31 && i % 4 === 0 ? { backgroundColor: 'rgb(218,218,218)' } : { backgroundColor: 'rgba(228,228,228,0.4)' }}
                  ></span>
                ))}
              </div>
              <div className="flex items-stretch gap-0 relative z-10">
                <div className="flex-1 min-w-0">
                  <OrderBlock order={conflict.priority_first} plan={priorityPlan} />
                </div>
                <div className="flex flex-col items-center justify-center flex-shrink-0 px-1 gap-0.5">
                  <svg className="h-3.5 w-3.5 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                  </svg>
                </div>
                <div className="flex-1 min-w-0">
                  <OrderBlock order={conflict.edf_first} plan={edfPlan} />
                </div>
              </div>
            </div>
          </div>

          {/* Swap indicator */}
          <div className="flex items-center gap-3 px-4 py-1.5 bg-amber-50 border-y border-amber-100">
            <svg className="h-4 w-4 text-amber-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
            </svg>
            <span className="text-[11px] font-semibold text-amber-700">EDF swapped queue position — earlier deadline takes precedence</span>
          </div>

          {/* EDF Order (what actually happened) */}
          <div className="px-4 py-3 bg-white/60">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-600">Actual Schedule</span>
              <span className="inline-flex items-center gap-1 text-[10px] text-emerald-500">
                <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
                applied
              </span>
            </div>
            {/* Single spanning timeline */}
            <div className="relative mb-2">
              <div className="flex items-center">
                <span className="w-1.5 h-1.5 rounded-full bg-gray-300 flex-shrink-0 z-10"></span>
                {edfPlan && <span className="ml-1.5 text-[9px] font-mono text-gray-400">{formatDate(edfPlan.starts_at)}</span>}
                <div className="flex-1 mx-1.5 h-px bg-gray-200"></div>
                {edfPlan && <span className="text-[9px] font-mono text-gray-400">{formatDate(edfPlan.ends_at)}</span>}
                <span className="w-1.5 h-1.5 rounded-full bg-gray-300 mx-1.5 flex-shrink-0 z-10"></span>
                <div className="flex-1 mx-0 h-px bg-gray-200"></div>
                {priorityPlan && <span className="text-[9px] font-mono text-gray-400">{formatDate(priorityPlan.ends_at)}</span>}
                <span className="w-1.5 h-1.5 rounded-full bg-gray-400 ml-1.5 flex-shrink-0 z-10"></span>
              </div>
            </div>
            <div className="relative py-4">
              {/* Background vertical lines */}
              <div className="absolute flex justify-between" style={{ top: 0, bottom: 0, left: '-6px', right: '-6px' }} aria-hidden="true">
                {Array.from({ length: 32 }).map((_, i) => (
                  <span key={i} className="w-px h-full"
                    style={ i > 0 && i < 31 && i % 4 === 0 ? { backgroundColor: 'rgb(218,218,218)' } : { backgroundColor: 'rgba(228,228,228,0.4)' }}
                  ></span>
                ))}
              </div>
              <div className="flex items-stretch gap-0 relative z-10">
                <div className="flex-1 min-w-0">
                  <OrderBlock order={conflict.edf_first} plan={edfPlan} />
                </div>
                <div className="flex flex-col items-center justify-center flex-shrink-0 px-1 gap-0.5">
                  <svg className="h-3.5 w-3.5 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                  </svg>
                </div>
                <div className="flex-1 min-w-0">
                  <OrderBlock order={conflict.priority_first} plan={priorityPlan} />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

/* ───────────────────────────────────────────────
   Main component
   ─────────────────────────────────────────────── */
export default function ScheduleLogTab({ schedule, productionLog, onRefresh }) {
  const [logFilter, setLogFilter] = useState('schedule')

  const filteredLog = logFilter === 'all'
    ? productionLog
    : productionLog.filter(e => e.category === 'schedule' || e.category === 'conflict')

  // Build a lookup: edf_first order_number → conflict object for fast matching
  const conflictByEdfOrder = {}
  ;(schedule?.conflicts || []).forEach(c => {
    conflictByEdfOrder[c.edf_first.order] = c
  })

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-slate-800 flex items-center justify-center">
            <svg className="h-4.5 w-4.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
          </div>
          <div>
            <h2 className="text-base font-semibold text-gray-900">Production Log</h2>
            <p className="text-xs text-gray-400">Schedule changes, conflicts &amp; events</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {/* Log filter toggle */}
          <div className="flex bg-gray-100 rounded-lg p-0.5">
            <button
              onClick={() => setLogFilter('schedule')}
              className={`text-[11px] font-medium px-2.5 py-1 rounded-md transition ${
                logFilter === 'schedule'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Schedule
            </button>
            <button
              onClick={() => setLogFilter('all')}
              className={`text-[11px] font-medium px-2.5 py-1 rounded-md transition ${
                logFilter === 'all'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Full System Log
            </button>
          </div>
          {schedule?.schedule_version && (
            <span className="text-[11px] font-semibold text-slate-500 bg-slate-100 rounded-md px-2.5 py-1 font-mono">
              v{schedule.schedule_version}
            </span>
          )}
          <span className="text-[11px] font-medium text-gray-400 bg-gray-50 rounded-md px-2.5 py-1">
            {filteredLog.length} entries
          </span>
          <button
            onClick={onRefresh}
            className="ml-1 p-1.5 rounded-md text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition"
            title="Refresh log"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>
      </div>

      {productionLog.length === 0 ? (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center">
          <div className="mx-auto w-14 h-14 rounded-full bg-gray-50 flex items-center justify-center mb-4">
            <svg className="h-7 w-7 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <p className="text-gray-400 text-sm">No log entries yet. Generate a schedule to start logging.</p>
        </div>
      ) : (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          {(() => {
            const grouped = []
            let currentVersion = Symbol('unset')
            let currentGroup = null
            filteredLog.forEach(entry => {
              const v = entry.schedule_version ?? 'none'
              if (v !== currentVersion) {
                currentVersion = v
                currentGroup = { version: v, entries: [] }
                grouped.push(currentGroup)
              }
              currentGroup.entries.push(entry)
            })

            return grouped.map((group, gi) => (
              <div key={gi}>
                {/* Version separator */}
                <div className="sticky top-0 z-10 px-5 py-2 bg-gray-50 border-b border-gray-200 flex items-center gap-2">
                  <span className="text-[11px] font-bold text-slate-600 font-mono bg-white border border-gray-200 rounded px-2 py-0.5">
                    {group.version === 'none' ? 'Events' : `v${group.version}`}
                  </span>
                  <div className="flex-1 h-px bg-gray-200"></div>
                  <span className="text-[10px] text-gray-400">
                    {group.entries.length} {group.entries.length === 1 ? 'event' : 'events'}
                  </span>
                </div>
                <div className="divide-y divide-gray-50">
                  {group.entries.map((entry) => {
                    const levelConfig = {
                      info:    { dot: 'bg-blue-500', badge: 'bg-blue-50 text-blue-600 ring-blue-500/20' },
                      warning: { dot: 'bg-amber-400', badge: 'bg-amber-50 text-amber-700 ring-amber-500/20' },
                      change:  { dot: 'bg-violet-500', badge: 'bg-violet-50 text-violet-600 ring-violet-500/20' },
                      error:   { dot: 'bg-red-500', badge: 'bg-red-50 text-red-600 ring-red-500/20' },
                    }
                    const cfg = levelConfig[entry.level] || levelConfig.info

                    // Resolve conflict data for conflict entries
                    const isConflict = entry.category === 'conflict'
                    const matchedConflict = isConflict ? conflictByEdfOrder[entry.order_number] : null

                    return (
                      <div key={entry.id} className="group px-5 py-3 hover:bg-gray-50/60 transition">
                        <div className="flex items-start gap-3">
                          {/* Timeline dot */}
                          <div className="flex flex-col items-center pt-1.5 flex-shrink-0">
                            <span className={`w-2 h-2 rounded-full ${cfg.dot} ring-4 ring-white`}></span>
                          </div>
                          {/* Content */}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-[13px] text-gray-900">{entry.title}</span>
                              <span className={`text-[10px] font-semibold rounded-md px-1.5 py-0.5 ring-1 ring-inset ${cfg.badge}`}>
                                {entry.level}
                              </span>
                              {entry.order_number && (
                                <span className="text-[11px] font-mono font-medium text-gray-400 bg-gray-100 rounded px-1.5 py-0.5">
                                  {entry.order_number}
                                </span>
                              )}
                              {matchedConflict && (
                                <span className="text-[11px] font-mono font-medium text-amber-500 bg-amber-50 rounded px-1.5 py-0.5">
                                  {matchedConflict.priority_first.order}
                                </span>
                              )}
                            </div>
                            {entry.detail && (
                              <p className="text-xs text-gray-500 mt-1 leading-relaxed">{entry.detail}</p>
                            )}
                            {/* Collapsible conflict visual */}
                            {matchedConflict && (
                              <ConflictVisual
                                conflict={matchedConflict}
                                plans={schedule?.production_plans}
                              />
                            )}
                          </div>
                          {/* Timestamp */}
                          <span className="text-[11px] text-gray-300 group-hover:text-gray-400 flex-shrink-0 whitespace-nowrap pt-0.5 transition font-mono">
                            {entry.timestamp && new Date(entry.timestamp).toLocaleTimeString('en-US', {
                              hour: '2-digit', minute: '2-digit', second: '2-digit'
                            })}
                          </span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            ))
          })()}
        </div>
      )}
    </div>
  )
}
