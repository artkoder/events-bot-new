-- Google AI shared rate-limit RPC rollout for partially migrated Supabase projects.
-- Safe/idempotent: additive only, no DROP/RENAME.
--
-- Why this migration:
-- - some projects already have google_ai_model_limits/google_ai_api_keys + legacy
--   finalize_google_ai_usage, but don't have google_ai_reserve/mark_sent/finalize
--   and request/counter tables.
-- - without these RPCs, cross-service RPM/TPM/RPD control is not enforced.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------------
-- 1) Make legacy core tables forward-compatible (if they already exist).
-- ---------------------------------------------------------------------------
ALTER TABLE IF EXISTS google_ai_model_limits
    ADD COLUMN IF NOT EXISTS tpm_reserve_extra INT NOT NULL DEFAULT 1000;

ALTER TABLE IF EXISTS google_ai_model_limits
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

ALTER TABLE IF EXISTS google_ai_api_keys
    ADD COLUMN IF NOT EXISTS account_name TEXT NULL;

ALTER TABLE IF EXISTS google_ai_api_keys
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

-- ---------------------------------------------------------------------------
-- 2) Core counter/audit tables required by new RPC.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS google_ai_usage_counters (
    id BIGSERIAL PRIMARY KEY,
    api_key_id UUID NOT NULL REFERENCES google_ai_api_keys(id),
    model TEXT NOT NULL,
    minute_bucket TIMESTAMPTZ NULL,
    day_bucket DATE NOT NULL,
    rpm_used INT NOT NULL DEFAULT 0,
    tpm_used INT NOT NULL DEFAULT 0,
    rpd_used INT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_google_ai_usage_counters_minute
    ON google_ai_usage_counters (api_key_id, model, minute_bucket)
    WHERE minute_bucket IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_google_ai_usage_counters_day
    ON google_ai_usage_counters (api_key_id, model, day_bucket)
    WHERE minute_bucket IS NULL;

CREATE INDEX IF NOT EXISTS idx_google_ai_usage_counters_minute_bucket
    ON google_ai_usage_counters (minute_bucket)
    WHERE minute_bucket IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_google_ai_usage_counters_day_bucket
    ON google_ai_usage_counters (day_bucket)
    WHERE minute_bucket IS NULL;

CREATE TABLE IF NOT EXISTS google_ai_requests (
    request_uid UUID PRIMARY KEY,
    consumer TEXT NOT NULL,
    account_name TEXT NULL,
    provider TEXT NOT NULL DEFAULT 'google',
    model TEXT NOT NULL,
    api_key_id UUID NULL REFERENCES google_ai_api_keys(id),
    minute_bucket TIMESTAMPTZ NULL,
    day_bucket DATE NULL,
    reserved_rpm INT NOT NULL DEFAULT 1,
    reserved_tpm INT NOT NULL,
    reserved_rpd INT NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'reserved',
    attempts INT NOT NULL DEFAULT 1,
    last_error_kind TEXT NULL,
    last_error_code TEXT NULL,
    last_error_message TEXT NULL,
    sent_at TIMESTAMPTZ NULL,
    finalized_at TIMESTAMPTZ NULL,
    usage_input_tokens INT NULL,
    usage_output_tokens INT NULL,
    usage_total_tokens INT NULL,
    meta JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_google_ai_requests_created
    ON google_ai_requests (created_at);

CREATE INDEX IF NOT EXISTS idx_google_ai_requests_consumer
    ON google_ai_requests (consumer, created_at);

CREATE INDEX IF NOT EXISTS idx_google_ai_requests_status
    ON google_ai_requests (status, updated_at);

CREATE TABLE IF NOT EXISTS google_ai_request_attempts (
    id BIGSERIAL PRIMARY KEY,
    request_uid UUID NOT NULL REFERENCES google_ai_requests(request_uid),
    attempt_no INT NOT NULL,
    status TEXT NOT NULL DEFAULT 'reserved',
    blocked_reason TEXT NULL,
    retry_after_ms INT NULL,
    api_key_id UUID NULL REFERENCES google_ai_api_keys(id),
    reserved_tpm INT NOT NULL,
    usage_input_tokens INT NULL,
    usage_output_tokens INT NULL,
    usage_total_tokens INT NULL,
    duration_ms INT NULL,
    provider_status TEXT NULL,
    provider_error_type TEXT NULL,
    provider_error_code TEXT NULL,
    provider_error_message TEXT NULL,
    meta JSONB NOT NULL DEFAULT '{}'::JSONB,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ NULL,
    CONSTRAINT google_ai_request_attempts_unique UNIQUE (request_uid, attempt_no)
);

CREATE INDEX IF NOT EXISTS idx_google_ai_request_attempts_started
    ON google_ai_request_attempts (started_at);

-- ---------------------------------------------------------------------------
-- 3) RPC: reserve -> mark_sent -> finalize
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION google_ai_reserve(
    p_request_uid UUID,
    p_attempt_no INT,
    p_consumer TEXT,
    p_account_name TEXT,
    p_model TEXT,
    p_reserved_tpm INT,
    p_candidate_key_ids UUID[] DEFAULT NULL
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_now TIMESTAMPTZ := timezone('utc', now());
    v_minute_bucket TIMESTAMPTZ := date_trunc('minute', v_now);
    v_day_bucket DATE := v_now::date;
    v_limits RECORD;
    v_key RECORD;
    v_minute_used RECORD;
    v_day_used RECORD;
    v_retry_after_ms INT;
    v_blocked_reason TEXT;
BEGIN
    SELECT * INTO v_limits FROM google_ai_model_limits WHERE model = p_model;
    IF v_limits IS NULL THEN
        RETURN jsonb_build_object(
            'ok', false,
            'blocked_reason', 'model_not_found',
            'message', 'Model not found in google_ai_model_limits'
        );
    END IF;

    IF EXISTS (
        SELECT 1 FROM google_ai_request_attempts
        WHERE request_uid = p_request_uid AND attempt_no = p_attempt_no
    ) THEN
        RETURN (
            SELECT jsonb_build_object(
                'ok', true,
                'api_key_id', r.api_key_id,
                'env_var_name', k.env_var_name,
                'key_alias', k.key_alias,
                'minute_bucket', r.minute_bucket,
                'day_bucket', r.day_bucket,
                'idempotent', true
            )
            FROM google_ai_requests r
            LEFT JOIN google_ai_api_keys k ON r.api_key_id = k.id
            WHERE r.request_uid = p_request_uid
        );
    END IF;

    FOR v_key IN
        SELECT * FROM google_ai_api_keys
        WHERE is_active = true
          AND (p_candidate_key_ids IS NULL OR id = ANY(p_candidate_key_ids))
        ORDER BY priority, id
    LOOP
        SELECT rpm_used, tpm_used INTO v_minute_used
        FROM google_ai_usage_counters
        WHERE api_key_id = v_key.id
          AND model = p_model
          AND minute_bucket = v_minute_bucket;

        IF v_minute_used IS NULL THEN
            v_minute_used.rpm_used := 0;
            v_minute_used.tpm_used := 0;
        END IF;

        SELECT rpd_used INTO v_day_used
        FROM google_ai_usage_counters
        WHERE api_key_id = v_key.id
          AND model = p_model
          AND day_bucket = v_day_bucket
          AND minute_bucket IS NULL;

        IF v_day_used IS NULL THEN
            v_day_used.rpd_used := 0;
        END IF;

        IF v_minute_used.rpm_used + 1 > v_limits.rpm THEN
            v_blocked_reason := 'rpm';
            v_retry_after_ms := (60 - EXTRACT(SECOND FROM v_now)::INT) * 1000;
            CONTINUE;
        END IF;

        IF v_minute_used.tpm_used + p_reserved_tpm > v_limits.tpm THEN
            v_blocked_reason := 'tpm';
            v_retry_after_ms := (60 - EXTRACT(SECOND FROM v_now)::INT) * 1000;
            CONTINUE;
        END IF;

        IF v_day_used.rpd_used + 1 > v_limits.rpd THEN
            v_blocked_reason := 'rpd';
            v_retry_after_ms := NULL;
            CONTINUE;
        END IF;

        INSERT INTO google_ai_usage_counters
            (api_key_id, model, minute_bucket, day_bucket, rpm_used, tpm_used)
        VALUES
            (v_key.id, p_model, v_minute_bucket, v_day_bucket, 1, p_reserved_tpm)
        ON CONFLICT (api_key_id, model, minute_bucket)
        WHERE minute_bucket IS NOT NULL
        DO UPDATE SET
            rpm_used = google_ai_usage_counters.rpm_used + 1,
            tpm_used = google_ai_usage_counters.tpm_used + p_reserved_tpm,
            updated_at = NOW();

        INSERT INTO google_ai_usage_counters
            (api_key_id, model, minute_bucket, day_bucket, rpd_used)
        VALUES
            (v_key.id, p_model, NULL, v_day_bucket, 1)
        ON CONFLICT (api_key_id, model, day_bucket)
        WHERE minute_bucket IS NULL
        DO UPDATE SET
            rpd_used = google_ai_usage_counters.rpd_used + 1,
            updated_at = NOW();

        INSERT INTO google_ai_requests (
            request_uid, consumer, account_name, model, api_key_id,
            minute_bucket, day_bucket, reserved_tpm, status
        ) VALUES (
            p_request_uid, p_consumer, p_account_name, p_model, v_key.id,
            v_minute_bucket, v_day_bucket, p_reserved_tpm, 'reserved'
        )
        ON CONFLICT (request_uid) DO NOTHING;

        INSERT INTO google_ai_request_attempts (
            request_uid, attempt_no, status, api_key_id, reserved_tpm
        ) VALUES (
            p_request_uid, p_attempt_no, 'reserved', v_key.id, p_reserved_tpm
        );

        RETURN jsonb_build_object(
            'ok', true,
            'api_key_id', v_key.id,
            'env_var_name', v_key.env_var_name,
            'key_alias', v_key.key_alias,
            'minute_bucket', v_minute_bucket,
            'day_bucket', v_day_bucket,
            'limits', jsonb_build_object('rpm', v_limits.rpm, 'tpm', v_limits.tpm, 'rpd', v_limits.rpd),
            'used_after', jsonb_build_object(
                'rpm', v_minute_used.rpm_used + 1,
                'tpm', v_minute_used.tpm_used + p_reserved_tpm,
                'rpd', v_day_used.rpd_used + 1
            )
        );
    END LOOP;

    RETURN jsonb_build_object(
        'ok', false,
        'blocked_reason', COALESCE(v_blocked_reason, 'no_keys'),
        'retry_after_ms', v_retry_after_ms,
        'minute_bucket', v_minute_bucket,
        'day_bucket', v_day_bucket
    );
END;
$$;

CREATE OR REPLACE FUNCTION google_ai_mark_sent(
    p_request_uid UUID,
    p_attempt_no INT
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE google_ai_requests
    SET sent_at = NOW(), status = 'sent', updated_at = NOW()
    WHERE request_uid = p_request_uid AND sent_at IS NULL;

    UPDATE google_ai_request_attempts
    SET status = 'sent'
    WHERE request_uid = p_request_uid AND attempt_no = p_attempt_no;
END;
$$;

CREATE OR REPLACE FUNCTION google_ai_finalize(
    p_request_uid UUID,
    p_attempt_no INT,
    p_usage_input_tokens INT,
    p_usage_output_tokens INT,
    p_usage_total_tokens INT,
    p_duration_ms INT,
    p_provider_status TEXT,
    p_error_type TEXT DEFAULT NULL,
    p_error_code TEXT DEFAULT NULL,
    p_error_message TEXT DEFAULT NULL
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    v_request RECORD;
    v_reserved_tpm INT;
    v_delta INT;
BEGIN
    SELECT * INTO v_request FROM google_ai_requests WHERE request_uid = p_request_uid;
    IF v_request IS NULL THEN
        RETURN;
    END IF;

    IF v_request.finalized_at IS NOT NULL THEN
        RETURN;
    END IF;

    SELECT reserved_tpm INTO v_reserved_tpm
    FROM google_ai_request_attempts
    WHERE request_uid = p_request_uid AND attempt_no = p_attempt_no;

    IF p_usage_total_tokens IS NOT NULL AND v_reserved_tpm IS NOT NULL THEN
        v_delta := p_usage_total_tokens - v_reserved_tpm;
        IF v_delta != 0 AND v_request.minute_bucket IS NOT NULL THEN
            UPDATE google_ai_usage_counters
            SET tpm_used = tpm_used + v_delta, updated_at = NOW()
            WHERE api_key_id = v_request.api_key_id
              AND model = v_request.model
              AND minute_bucket = v_request.minute_bucket;
        END IF;
    END IF;

    UPDATE google_ai_requests
    SET
        status = CASE WHEN p_error_type IS NULL THEN 'succeeded' ELSE 'failed_provider' END,
        finalized_at = NOW(),
        usage_input_tokens = p_usage_input_tokens,
        usage_output_tokens = p_usage_output_tokens,
        usage_total_tokens = p_usage_total_tokens,
        last_error_kind = CASE WHEN p_error_type IS NOT NULL THEN 'provider' ELSE NULL END,
        last_error_code = p_error_code,
        last_error_message = p_error_message,
        updated_at = NOW()
    WHERE request_uid = p_request_uid;

    UPDATE google_ai_request_attempts
    SET
        status = CASE WHEN p_error_type IS NULL THEN 'succeeded' ELSE 'failed_provider' END,
        usage_input_tokens = p_usage_input_tokens,
        usage_output_tokens = p_usage_output_tokens,
        usage_total_tokens = p_usage_total_tokens,
        duration_ms = p_duration_ms,
        provider_status = p_provider_status,
        provider_error_type = p_error_type,
        provider_error_code = p_error_code,
        provider_error_message = p_error_message,
        completed_at = NOW()
    WHERE request_uid = p_request_uid AND attempt_no = p_attempt_no;
END;
$$;

GRANT EXECUTE ON FUNCTION google_ai_reserve(UUID, INT, TEXT, TEXT, TEXT, INT, UUID[]) TO service_role;
GRANT EXECUTE ON FUNCTION google_ai_mark_sent(UUID, INT) TO service_role;
GRANT EXECUTE ON FUNCTION google_ai_finalize(UUID, INT, INT, INT, INT, INT, TEXT, TEXT, TEXT, TEXT) TO service_role;

-- Keep model seed additive-only (works even if model has no UNIQUE index).
INSERT INTO google_ai_model_limits (model, rpm, tpm, rpd)
SELECT s.model, s.rpm, s.tpm, s.rpd
FROM (
    VALUES
        ('gemma-3-27b', 30, 15000, 14400),
        ('gemini-2.5-flash', 5, 250000, 20),
        ('gemma-3-4b', 30, 15000, 14400),
        ('gemma-3-12b', 30, 15000, 14400),
        ('gemma-3-1b', 30, 15000, 14400)
) AS s(model, rpm, tpm, rpd)
WHERE NOT EXISTS (
    SELECT 1 FROM google_ai_model_limits m WHERE m.model = s.model
);

COMMIT;
