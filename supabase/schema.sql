-- ============================================================
-- NetSentinel AI — Phase 1 Supabase Schema
-- ============================================================
-- PURPOSE:
--   This schema is PROPOSED only. It is not yet applied to any
--   real Supabase project. It will be applied after Phase 1 is
--   fully stable and Supabase credentials are configured.
--
-- HOW TO APPLY:
--   1. Create a Supabase project at https://supabase.com
--   2. Copy SUPABASE_URL and SUPABASE_KEY from project Settings → API
--   3. Add them to your .env file (see .env.example)
--   4. Open the Supabase SQL Editor and paste this file
--   5. Execute to create the table
--   6. The application will automatically use Supabase when credentials
--      are present, and fall back to local/CSV mode when they are absent.
-- ============================================================

-- Enable UUID extension (usually enabled by default on Supabase)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- Table: network_traffic
-- Stores normalized network flow / traffic metadata records.
-- Raw packet payloads are NEVER stored (privacy + storage).
-- ============================================================
CREATE TABLE IF NOT EXISTS network_traffic (
    -- Unique record identifier
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- When this flow was observed (ISO 8601 timestamp)
    timestamp       TIMESTAMPTZ NOT NULL,

    -- Network endpoints
    source_ip       INET NOT NULL,
    destination_ip  INET NOT NULL,
    source_port     SMALLINT,               -- NULL for protocols like ICMP
    destination_port SMALLINT,              -- NULL for protocols like ICMP

    -- Network protocol (TCP, UDP, ICMP, etc.)
    protocol        VARCHAR(16) NOT NULL,

    -- Flow aggregation metrics
    packet_count    INTEGER NOT NULL DEFAULT 1 CHECK (packet_count > 0),
    byte_count      BIGINT  NOT NULL DEFAULT 0 CHECK (byte_count >= 0),

    -- Flow duration in seconds (NULL if instantaneous/unknown)
    duration        NUMERIC(12, 4),

    -- Calculated bytes-per-second rate (NULL if duration is 0 or unknown)
    traffic_rate    NUMERIC(16, 4),

    -- Metadata: identifies where this record came from
    -- 'csv'  = imported from a CSV file
    -- 'live' = captured from a live network interface
    source_type     VARCHAR(16) NOT NULL DEFAULT 'csv',

    -- When this record was ingested into the database
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Indexes for common query patterns
-- ============================================================

-- Timeline queries (traffic over time)
CREATE INDEX IF NOT EXISTS idx_traffic_timestamp
    ON network_traffic (timestamp DESC);

-- Protocol distribution queries
CREATE INDEX IF NOT EXISTS idx_traffic_protocol
    ON network_traffic (protocol);

-- Top source IP queries
CREATE INDEX IF NOT EXISTS idx_traffic_source_ip
    ON network_traffic (source_ip);

-- Top destination IP queries
CREATE INDEX IF NOT EXISTS idx_traffic_destination_ip
    ON network_traffic (destination_ip);

-- Source type filtering
CREATE INDEX IF NOT EXISTS idx_traffic_source_type
    ON network_traffic (source_type);

-- ============================================================
-- Comments
-- ============================================================
COMMENT ON TABLE network_traffic IS
    'NetSentinel AI Phase 1: normalized network flow metadata. '
    'Raw packet payloads are never stored.';

COMMENT ON COLUMN network_traffic.source_port IS
    'Source port number. NULL for connectionless protocols like ICMP.';

COMMENT ON COLUMN network_traffic.destination_port IS
    'Destination port number. NULL for connectionless protocols like ICMP.';

COMMENT ON COLUMN network_traffic.traffic_rate IS
    'Calculated as byte_count / duration (bytes per second). NULL when duration is 0 or unknown.';

COMMENT ON COLUMN network_traffic.source_type IS
    'Origin of this record: csv (imported) or live (real-time capture).';
