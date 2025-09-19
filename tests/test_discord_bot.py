import json

from src.discord_bot import DiscordMessageBuilder, SignalEnvelope, TierConfig, WebhookTarget


def _make_tier(detail_level: str) -> TierConfig:
    return TierConfig(
        name=detail_level,
        queue=f"queue:{detail_level}",
        detail_level=detail_level,
        mention='@everyone',
        webhook=WebhookTarget(
            key=f"{detail_level}-key",
            url="https://discord.test/webhook",
            channel_id=123,
            label=f"{detail_level.title()} Channel",
            mention_role_id=None,
            mention_override=None,
        ),
    )


def _base_payload() -> dict:
    return {
        'id': 'sig-123',
        'symbol': 'SPY',
        'side': 'LONG',
        'strategy': '0DTE',
        'confidence': 87,
        'action_type': 'ENTRY',
        'entry': 4.25,
        'stop': 3.25,
        'targets': [6.5],
        'position_notional': 12500,
        'reasons': ['Flow imbalance', 'Dealer positioning'],
        'contract': {
            'type': 'option',
            'expiry': '20240426',
            'strike': 455,
            'right': 'C',
            'symbol': 'SPY',
        },
        'execution': {
            'status': 'FILLED',
            'avg_fill_price': 4.25,
            'filled_quantity': 5,
            'executed_at': '2024-04-26T15:34:12',
        },
        'ts': 1714148052000,
    }


def test_signal_envelope_defaults_when_missing_id():
    payload = _base_payload()
    payload.pop('id')
    envelope = SignalEnvelope.from_dict(payload, 'premium')

    assert envelope.id.startswith('SPY:')
    assert envelope.action_type == 'ENTRY'
    assert envelope.targets == [6.5]
    assert envelope.execution_status == 'FILLED'


def test_premium_entry_message_contains_contract_and_risk_fields():
    envelope = SignalEnvelope.from_dict(_base_payload(), 'premium')
    builder = DiscordMessageBuilder({'colors': {'long': 0x00FF00}})
    tier = _make_tier('premium')

    message = builder.build(envelope, tier)

    assert 'Premium fill' in message.content
    field_names = [field['name'] for field in message.embeds[0]['fields']]
    assert 'Contract' in field_names
    assert 'Risk Plan' in field_names
    assert any('Flow imbalance' in field.get('value', '') for field in message.embeds[0]['fields'])


def test_basic_entry_layout_matches_spec():
    payload = _base_payload()
    payload.update({
        'confidence_band': 'HIGH',
        'execution': {'status': 'FILLED', 'avg_fill_price': 4.25, 'filled_quantity': 5, 'executed_at': '2024-04-30T14:30:00'},
    })
    envelope = SignalEnvelope.from_dict(payload, 'basic')
    builder = DiscordMessageBuilder({
        'basic_rationale_tags': {
            'flow': 'Flow alignment',
            'volatility': 'Volatility check',
            'structure': 'Structure aligned',
        },
        'basic_upgrade_text': 'Premium members receive instant alerts.',
    })
    tier = _make_tier('basic')

    message = builder.build(envelope, tier)
    embed = message.embeds[0]

    assert embed['title'] == 'SPY Long Entry'
    field_names = [field['name'] for field in embed['fields']]
    assert field_names == ['Contract', 'Execution', 'Risk Plan', 'Confidence', 'Drivers', 'Upgrade']
    values = {field['name']: field['value'] for field in embed['fields']}
    assert values['Contract'] == 'SPY 26 Apr 24 455.0C'
    assert values['Execution'] == 'Fill $4.25 · Size 5'
    assert values['Risk Plan'] == 'Entry $4.25 · Stop $3.25 · Target $6.50'
    assert values['Confidence'] == 'HIGH'
    assert values['Drivers'] == 'Flow alignment • Volatility check • Structure aligned'
    assert values['Upgrade'] == 'Premium members receive instant alerts.'
    assert message.content.strip() == '@everyone'
    assert embed['footer']['text'].endswith('AM') or embed['footer']['text'].endswith('PM')


def test_basic_exit_message_redacts_realized_pnl():
    payload = _base_payload()
    payload.update({
        'action_type': 'EXIT',
        'execution': {'status': 'CLOSED', 'executed_at': '2024-04-26T16:03:00'},
        'lifecycle': {
            'result': 'WIN',
            'realized_pnl': 1850.0,
            'return_pct': 0.148,
            'holding_period_minutes': 42.5,
        },
    })
    envelope = SignalEnvelope.from_dict(payload, 'basic')
    builder = DiscordMessageBuilder({})
    tier = _make_tier('basic')

    message = builder.build(envelope, tier)

    serialized = json.dumps(message.embeds)
    assert 'Realized' not in serialized
    assert 'WIN' in serialized
    assert any(field['name'] == 'Upgrade' for field in message.embeds[0]['fields'])


def test_teaser_message_contains_cta():
    payload = _base_payload()
    payload.update({
        'action_type': 'ENTRY',
        'sentiment': 'bullish',
        'message': 'Bullish flow spotted on SPY',
    })
    envelope = SignalEnvelope.from_dict(payload, 'free')
    builder = DiscordMessageBuilder({
        'free_upgrade_text': 'Upgrade for full deets.',
        'teaser_cta': 'Upgrade for full deets.',
    })
    tier = _make_tier('teaser')

    message = builder.build(envelope, tier)
    embed = message.embeds[0]

    assert message.content.strip() == '@everyone'
    assert embed['description'] == 'Upgrade for full deets.'
    fields = {field['name']: field['value'] for field in embed['fields']}
    assert fields['Contract'] == 'SPY 26 Apr 24 455.0C'
    assert fields['Execution'] == 'Fill $4.25'
    assert fields['Risk Plan'] == 'Entry $4.25 · Stop $3.25'
