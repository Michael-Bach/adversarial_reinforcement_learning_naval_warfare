--[[
cmano_bridge.lua — CMANO Lua bridge for adversarial RL training

Runs inside Command: Modern Operations (Steam edition) as a Lua script.
Communicates with the Python training process via file-based IPC.

State channel  : ScenEdit_ExportScenarioToXML() → state.xml  (parsed by pycmo.FeaturesFromSteam)
                 step/terminal metadata           → state_meta.json
Action channel : action.json  (JSON from Python, executed here as ScenEdit calls)

Installation
------------
1. Open a blank scenario in CMO.
2. Scenario Editor → Events → Add Event:
     Trigger  = "Scenario Loaded"
     Action   = Lua Script:
         dofile([[C:\path\to\cmano_bridge.lua]])
         CMATNOBridgeInit()
3. Add a second Event:
     Trigger  = "Regular Time" interval → step_seconds (default 30 s)
     Action   = Lua Script:  CMATNOBridgeStep()
4. Save as bootstrap.scen.
5. Launch: cmo.exe bootstrap.scen /autorun

Set the CMANO_BRIDGE_DIR environment variable on the CMO host to match
bridge_dir in your Python config (default: C:\cmano_bridge).
For cross-machine use, point this at a network share visible to both hosts.
--]]

-- ============================================================
-- Configuration
-- ============================================================

local BRIDGE_DIR = os.getenv("CMANO_BRIDGE_DIR") or [[C:\cmano_bridge]]

-- ============================================================
-- Module-level state
-- ============================================================

local _config            = nil
local _step              = 0
local _episode           = 0
local _time_comp         = 300
local _max_steps         = 500
local _episode_start_time = 0

-- Initial unit config keyed by unit_id, for in-episode reset.
local _initial_alice = {}
local _initial_bob   = {}
local _alice_ids     = {}   -- insertion-ordered unit_id lists
local _bob_ids       = {}

-- ============================================================
-- JSON (encoder only — Python decodes; Lua only encodes metadata)
-- ============================================================

local json_encode
do
    local ok = pcall(function()
        if JSON and JSON.encode then
            json_encode = function(t) return JSON:encode(t) end
        end
    end)
    if not json_encode then
        local function enc(v)
            local t = type(v)
            if t == "nil"       then return "null"
            elseif t == "boolean" then return tostring(v)
            elseif t == "number"  then
                if v ~= v then return "null" end
                return string.format("%.8g", v)
            elseif t == "string" then
                return '"' .. v:gsub('\\','\\\\'):gsub('"','\\"')
                             :gsub('\n','\\n'):gsub('\r','\\r') .. '"'
            elseif t == "table" then
                if #v > 0 then
                    local p = {}
                    for _, item in ipairs(v) do p[#p+1] = enc(item) end
                    return "[" .. table.concat(p, ",") .. "]"
                else
                    local p = {}
                    for k, val in pairs(v) do
                        p[#p+1] = enc(tostring(k)) .. ":" .. enc(val)
                    end
                    return "{" .. table.concat(p, ",") .. "}"
                end
            end
            return "null"
        end
        json_encode = enc
    end
end

-- JSON decoder (needed only for config and action messages from Python)
local json_decode
do
    local ok, cj = pcall(require, "cjson")
    if ok then
        json_decode = cj.decode
    else
        local ok2 = pcall(function()
            if JSON and JSON.decode then
                json_decode = function(s) return JSON:decode(s) end
            end
        end)
        if not json_decode then
            json_decode = function(_)
                error("No JSON decoder in this CMO version. Install cjson.dll or upgrade CMO.")
            end
        end
    end
end

-- ============================================================
-- File helpers
-- ============================================================

local function _p(name) return BRIDGE_DIR .. "\\" .. name end

local function fileExists(path)
    local f = io.open(path, "r")
    if f then f:close() return true end
    return false
end

local function readFile(path)
    local f = assert(io.open(path, "r"), "cannot open " .. path)
    local s = f:read("*a"); f:close(); return s
end

local function writeFile(path, content)
    local f = assert(io.open(path, "w"), "cannot write " .. path)
    f:write(content); f:close()
end

local function waitForFlag(name, timeout_sec)
    local deadline = os.time() + timeout_sec
    local path = _p(name)
    while not fileExists(path) do
        if os.time() >= deadline then return false end
        -- CPU spin: acceptable because the scenario is paused
    end
    return true
end

local function clearFlag(name) os.remove(_p(name)) end

-- ============================================================
-- Sensor helper
-- ============================================================

local function addOmniscientSensor(unit_name)
    -- Add a long-range radar so both sides always have contacts for all
    -- opponent units (required for ScenEdit_AttackContact).
    -- Replace DBID=1 with a real CMO sensor database ID if needed.
    pcall(ScenEdit_AddMountToUnit, {
        UnitName  = unit_name,
        MountName = "RL-Omniscient-Radar",
        DBID      = 1,
    })
end

-- ============================================================
-- Scenario building
-- ============================================================

local function buildSide(side_name, units_cfg, initial_store, id_list)
    pcall(ScenEdit_AddSide, {name = side_name})
    for _, u in ipairs(units_cfg) do
        local ok, err = pcall(ScenEdit_AddUnit, {
            side      = side_name,
            type      = "Ship",
            unitname  = u.unit_id,
            classname = u.cmano_classname or "FFG-7 Oliver Hazard Perry",
            dbid      = u.cmano_dbid,
            lat       = u.lat,
            lon       = u.lon,
            speed     = u.initial_speed_kts or 0,
            heading   = u.heading_deg or 0,
        })
        if not ok then
            writeFile(_p("bridge_error.txt"),
                "AddUnit failed [" .. u.unit_id .. "]: " .. tostring(err))
        end
        addOmniscientSensor(u.unit_id)
        initial_store[u.unit_id] = {
            lat         = u.lat,
            lon         = u.lon,
            heading_deg = u.heading_deg   or 0,
            speed_kts   = u.initial_speed_kts or 0,
            classname   = u.cmano_classname or "FFG-7 Oliver Hazard Perry",
            dbid        = u.cmano_dbid,
        }
        id_list[#id_list+1] = u.unit_id
    end
end

local function buildScenario(cfg)
    _time_comp = cfg.time_compression or 300
    _max_steps = cfg.max_steps        or 500
    ScenEdit_SetTimeCompression(0)
    buildSide("Alice", cfg.alice_units, _initial_alice, _alice_ids)
    buildSide("Bob",   cfg.bob_units,   _initial_bob,   _bob_ids)
    ScenEdit_SetTimeCompression(_time_comp)
    _episode_start_time = ScenEdit_CurrentTime()
end

-- ============================================================
-- State export — two modes
--
-- Fast path  (xml_needed = false):
--   Per-unit ScenEdit_GetUnit calls → state_fast.json
--   Python reads this directly; no XML parsing required.
--   Used on normal steps where no fire action occurred.
--
-- Full path  (xml_needed = true):
--   ScenEdit_ExportScenarioToXML() → state.xml
--   Used when fire=true on any unit (contact GUIDs needed for next attack)
--   or on the post-reset step (Python builds initial contact maps).
-- ============================================================

local function exportStateFast(terminal, reason)
    -- Collect unit states with ScenEdit_GetUnit (microseconds per unit vs
    -- hundreds of milliseconds for full XML export).
    local function collectSide(id_list, side_name)
        local units = {}
        for _, uid in ipairs(id_list) do
            local ok, obj = pcall(ScenEdit_GetUnit, {Name = uid, FromSide = side_name})
            if ok and obj ~= nil then
                units[#units+1] = {
                    unit_id     = uid,
                    lat         = tonumber(obj.latitude)  or 0,
                    lon         = tonumber(obj.longitude) or 0,
                    heading_deg = tonumber(obj.heading)   or 0,
                    speed_kts   = tonumber(obj.speed)     or 0,
                    damage_pct  = tonumber(obj.damage)    or 0,
                }
            else
                units[#units+1] = {unit_id = uid, dead = true}
            end
        end
        return units
    end

    local payload = {
        mode              = "fast",
        step              = _step,
        game_time_elapsed = ScenEdit_CurrentTime() - _episode_start_time,
        terminal          = terminal or false,
        terminal_reason   = reason  or "",
        alice_units       = collectSide(_alice_ids, "Alice"),
        bob_units         = collectSide(_bob_ids,   "Bob"),
    }
    writeFile(_p("state_fast.json"), json_encode(payload))
    -- state.xml is intentionally NOT written on the fast path.
    -- Python checks state_meta.json for mode="fast" and reads state_fast.json.
    writeFile(_p("state_meta.json"), json_encode({
        mode              = "fast",
        step              = _step,
        game_time_elapsed = payload.game_time_elapsed,
        terminal          = terminal or false,
        terminal_reason   = reason  or "",
    }))
end

local function exportStateXML(terminal, reason)
    -- Full XML export — slow but carries contact GUIDs for weapon targeting.
    local ok, xml_str = pcall(ScenEdit_ExportScenarioToXML)
    if ok and type(xml_str) == "string" and #xml_str > 0 then
        writeFile(_p("state.xml"), xml_str)
    else
        writeFile(_p("bridge_error.txt"),
            "ScenEdit_ExportScenarioToXML failed: " .. tostring(xml_str))
        writeFile(_p("state.xml"), "<Scenario/>")
    end
    writeFile(_p("state_meta.json"), json_encode({
        mode              = "full",
        step              = _step,
        game_time_elapsed = ScenEdit_CurrentTime() - _episode_start_time,
        terminal          = terminal or false,
        terminal_reason   = reason  or "",
    }))
end

-- exportState picks the right mode.
-- force_full=true on post-reset step and when any fire action was issued.
local function exportState(terminal, reason, force_full)
    if force_full then
        exportStateXML(terminal, reason)
    else
        exportStateFast(terminal, reason)
    end
end

-- ============================================================
-- Terminal condition check (uses ScenEdit_GetUnit for speed)
-- ============================================================

local function getUnitAlive(unit_id, side_name)
    local ok, obj = pcall(ScenEdit_GetUnit, {Name = unit_id, FromSide = side_name})
    if not ok or obj == nil then return false end
    local damage = tonumber(obj.damage) or 0
    return damage < 100
end

local function checkTerminal()
    local alice_alive, bob_alive = false, false
    for _, uid in ipairs(_alice_ids) do
        if getUnitAlive(uid, "Alice") then alice_alive = true; break end
    end
    for _, uid in ipairs(_bob_ids) do
        if getUnitAlive(uid, "Bob") then bob_alive = true; break end
    end
    if not alice_alive then return true, "alice_dead" end
    if not bob_alive   then return true, "bob_dead"   end
    if _step >= _max_steps then return true, "max_steps" end
    return false, ""
end

-- ============================================================
-- Action application
-- ============================================================

-- applyFleetActions returns true if any unit fired (triggers full XML export next step).
local function applyFleetActions(actions, initial_store)
    local any_fire = false
    for _, a in ipairs(actions) do
        local uid = a.unit_id
        if initial_store[uid] then
            pcall(ScenEdit_SetUnit, {
                Name    = uid,
                Speed   = tonumber(a.speed_kts)   or 0,
                Heading = tonumber(a.heading_deg) or 0,
            })
            -- target_id is a contact GUID (from pycmo Contact.ID) or unit name.
            -- ScenEdit_AttackContact accepts both in most CMO versions.
            if a.fire and a.target_id and a.target_id ~= "" then
                pcall(ScenEdit_AttackContact, {
                    attacker = uid,
                    target   = a.target_id,
                })
                any_fire = true
            end
        end
    end
    return any_fire
end

-- applyActions returns true if any unit fired this step.
local function applyActions(action_msg)
    local fire_A = applyFleetActions(action_msg.alice_actions or {}, _initial_alice)
    local fire_B = applyFleetActions(action_msg.bob_actions   or {}, _initial_bob)
    return fire_A or fire_B
end

-- ============================================================
-- Episode reset
-- ============================================================

local function resetEpisode(episode_num)
    _episode = episode_num
    _step    = 0
    ScenEdit_SetTimeCompression(0)

    local function rebuildSide(initial_store, id_list, side_name)
        for _, uid in ipairs(id_list) do
            pcall(ScenEdit_DeleteUnit, {Name = uid})
            local init = initial_store[uid]
            local ok, err = pcall(ScenEdit_AddUnit, {
                side      = side_name,
                type      = "Ship",
                unitname  = uid,
                classname = init.classname,
                dbid      = init.dbid,
                lat       = init.lat,
                lon       = init.lon,
                speed     = init.speed_kts,
                heading   = init.heading_deg,
            })
            if not ok then
                writeFile(_p("bridge_error.txt"),
                    "Reset rebuild failed [" .. uid .. "]: " .. tostring(err))
            end
            addOmniscientSensor(uid)
        end
    end

    rebuildSide(_initial_alice, _alice_ids, "Alice")
    rebuildSide(_initial_bob,   _bob_ids,   "Bob")
    _episode_start_time = ScenEdit_CurrentTime()
end

-- ============================================================
-- Step handler (called by CMO Regular Time event)
-- ============================================================

-- _next_step_full: when true the next state export will use full XML.
-- Set after any step where fire actions were issued, so Python gets fresh
-- contact GUIDs for the subsequent attack command.
local _next_step_full = false

function CMATNOBridgeStep()
    if not _config then return end

    -- Episode reset takes priority
    if fileExists(_p("reset.flag")) then
        clearFlag("reset.flag")
        local ok, reset_msg = pcall(json_decode, readFile(_p("reset.json")))
        resetEpisode(ok and reset_msg.episode or (_episode + 1))
        _next_step_full = false

        -- Always export full XML on post-reset step so Python builds initial contact maps.
        exportState(false, "", true)
        writeFile(_p("state.flag"), "1")
        return
    end

    _step = _step + 1
    ScenEdit_SetTimeCompression(0)

    local terminal, reason = checkTerminal()
    -- Use full XML if the previous step fired weapons, or on terminal (Python
    -- may need contact state for the final observation).
    local force_full = _next_step_full or terminal
    exportState(terminal, reason, force_full)
    _next_step_full = false
    writeFile(_p("state.flag"), "1")   -- signal Python AFTER all state files written

    if terminal then
        return  -- stay paused; Python sends reset.flag to start next episode
    end

    -- Wait for Python's action (30 real-second timeout)
    if not waitForFlag("action.flag", 30) then
        ScenEdit_SetTimeCompression(_time_comp)
        return
    end

    local ok, action_msg = pcall(json_decode, readFile(_p("action.json")))
    clearFlag("action.flag")

    if ok then
        local any_fire = applyActions(action_msg)
        -- If weapons were just fired, emit a full XML state on the *next* step so
        -- Python can refresh contact GUIDs before issuing the next attack.
        _next_step_full = any_fire
    else
        writeFile(_p("bridge_error.txt"), "Action decode error: " .. tostring(action_msg))
    end

    ScenEdit_SetTimeCompression(_time_comp)
end

-- ============================================================
-- Initialisation (called from Scenario Loaded event)
-- ============================================================

function CMATNOBridgeInit()
    if not waitForFlag("config.flag", 120) then
        writeFile(_p("bridge_error.txt"),
            "CMATNOBridgeInit: timed out waiting for config.flag")
        return
    end
    clearFlag("config.flag")

    local ok, cfg = pcall(json_decode, readFile(_p("config.json")))
    if not ok then
        writeFile(_p("bridge_error.txt"), "Config parse error: " .. tostring(cfg))
        return
    end
    _config = cfg

    buildScenario(cfg)
    writeFile(_p("ready.flag"), "1")
end

-- ============================================================
-- Shutdown
-- ============================================================

function CMATNOBridgeShutdown()
    if fileExists(_p("shutdown.flag")) then
        clearFlag("shutdown.flag")
        ScenEdit_SetTimeCompression(0)
    end
end
