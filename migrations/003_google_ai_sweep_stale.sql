-- Sweep stale Google AI reservations that never reached mark_sent/finalize.
-- Safe policy:
--   * only compensate rows with status='reserved' AND sent_at IS NULL
--   * compensate counters down to >= 0
--   * mark request/attempt rows as status='stale'

CREATE OR REPLACE FUNCTION google_ai_sweep_stale(
    p_older_than_minutes INT DEFAULT 30,
    p_limit INT DEFAULT 500
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_now TIMESTAMPTZ := timezone('utc', now());
    v_cutoff TIMESTAMPTZ := v_now - make_interval(mins => GREATEST(1, p_older_than_minutes));
    v_limit INT := GREATEST(1, LEAST(p_limit, 5000));
    v_req RECORD;
    v_swept INT := 0;
    v_ids UUID[] := ARRAY[]::UUID[];
BEGIN
    FOR v_req IN
        SELECT
            r.request_uid,
            r.api_key_id,
            r.model,
            r.minute_bucket,
            r.day_bucket,
            COALESCE(r.reserved_rpm, 1) AS reserved_rpm,
            COALESCE(r.reserved_tpm, 0) AS reserved_tpm,
            COALESCE(r.reserved_rpd, 1) AS reserved_rpd
        FROM google_ai_requests r
        WHERE r.status = 'reserved'
          AND r.sent_at IS NULL
          AND r.finalized_at IS NULL
          AND r.created_at < v_cutoff
        ORDER BY r.created_at
        LIMIT v_limit
        FOR UPDATE SKIP LOCKED
    LOOP
        IF v_req.minute_bucket IS NOT NULL THEN
            UPDATE google_ai_usage_counters
            SET
                rpm_used = GREATEST(0, COALESCE(rpm_used, 0) - v_req.reserved_rpm),
                tpm_used = GREATEST(0, COALESCE(tpm_used, 0) - v_req.reserved_tpm),
                updated_at = NOW()
            WHERE api_key_id = v_req.api_key_id
              AND model = v_req.model
              AND minute_bucket = v_req.minute_bucket;
        END IF;

        IF v_req.day_bucket IS NOT NULL THEN
            UPDATE google_ai_usage_counters
            SET
                rpd_used = GREATEST(0, COALESCE(rpd_used, 0) - v_req.reserved_rpd),
                updated_at = NOW()
            WHERE api_key_id = v_req.api_key_id
              AND model = v_req.model
              AND day_bucket = v_req.day_bucket
              AND minute_bucket IS NULL;
        END IF;

        UPDATE google_ai_requests
        SET
            status = 'stale',
            last_error_kind = 'stale',
            last_error_code = 'reserve_not_sent_timeout',
            last_error_message = 'swept stale reserved (sent_at is null)',
            finalized_at = COALESCE(finalized_at, NOW()),
            updated_at = NOW()
        WHERE request_uid = v_req.request_uid;

        UPDATE google_ai_request_attempts
        SET
            status = 'stale',
            provider_error_type = COALESCE(provider_error_type, 'stale'),
            provider_error_code = COALESCE(provider_error_code, 'reserve_not_sent_timeout'),
            provider_error_message = COALESCE(provider_error_message, 'swept stale reserved (sent_at is null)'),
            completed_at = COALESCE(completed_at, NOW())
        WHERE request_uid = v_req.request_uid
          AND status = 'reserved';

        v_swept := v_swept + 1;
        v_ids := array_append(v_ids, v_req.request_uid);
    END LOOP;

    RETURN jsonb_build_object(
        'ok', true,
        'swept', v_swept,
        'cutoff', v_cutoff,
        'request_uids', v_ids
    );
END;
$$;

GRANT EXECUTE ON FUNCTION google_ai_sweep_stale(INT, INT) TO service_role;
