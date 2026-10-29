import { formatSyncedTime } from '../lib/format';

function LiveIndicator({ lastSynced }) {
  return (
    <div className="inline-flex items-center gap-1.5 text-[10px] text-gray-500 uppercase tracking-wider">
      <span className="relative flex h-1.5 w-1.5 shrink-0">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-70" />
        <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-400" />
      </span>
      <span>Last synced: {formatSyncedTime(lastSynced)}</span>
    </div>
  );
}

export default LiveIndicator;
