import { useState } from 'react'

export default function ScheduleLogTab({ schedule, productionLog, onRefresh }) {
  const [logFilter, setLogFilter] = useState('schedule')

  const filteredLog = logFilter === 'all'
    ? productionLog
    : productionLog.filter(e => e.category === 'schedule' || e.category === 'conflict')

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

                    return (
                      <div key={entry.id} className="group px-5 py-3 hover:bg-gray-50/60 transition flex items-start gap-3">
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
                          </div>
                          {entry.detail && (
                            <p className="text-xs text-gray-500 mt-1 leading-relaxed">{entry.detail}</p>
                          )}
                        </div>
                        {/* Timestamp */}
                        <span className="text-[11px] text-gray-300 group-hover:text-gray-400 flex-shrink-0 whitespace-nowrap pt-0.5 transition font-mono">
                          {entry.timestamp && new Date(entry.timestamp).toLocaleTimeString('en-US', {
                            hour: '2-digit', minute: '2-digit', second: '2-digit'
                          })}
                        </span>
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
