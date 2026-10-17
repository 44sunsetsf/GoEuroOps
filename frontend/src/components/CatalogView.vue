<template>
  <section class="page page-scroll page-catalog">
    <div class="page-heading">
      <div class="heading-copy">
        <h1>{{ t('服务价目') }}</h1>
        <p>
          {{ t('助手报价、退款计算和知识库都读取同一份业务目录') }}
          <code>backend/business/catalog.yaml</code>{{ t('。价格可以在这里直接改，保存后立即生效；改过的价格旁边会标出默认价。') }}
        </p>
      </div>
      <div class="heading-actions">
        <template v-if="editing">
          <button @click="savePrices" :disabled="saving">{{ saving ? t('保存中') : t('保存价格') }}</button>
          <button class="quiet-button" @click="editing = false" :disabled="saving">{{ t('取消') }}</button>
        </template>
        <button v-else class="quiet-button" @click="startEditing" :disabled="readonly || !catalog" :title="readonly ? t('演示模式只读，这个操作只对站长开放') : ''">{{ t('改价格') }}</button>
      </div>
      <div v-if="catalog" class="count-display"><strong>{{ catalog.services.length }}</strong><span>{{ t('项服务') }} · {{ catalog.catalog_version }}</span></div>
    </div>

    <div v-if="error" class="inline-error">{{ error }}</div>

    <template v-if="catalog">
      <section v-for="group in groups" :key="group.category" class="catalog-group">
        <h2 class="group-title">{{ group.label }}</h2>
        <div class="service-grid">
          <article v-for="service in group.services" :key="service.sku" class="service-card">
            <header>
              <div>
                <strong>{{ service.name }}</strong>
                <code>{{ service.sku }}</code>
              </div>
              <div class="price-tag">
                <template v-if="editing">
                  <input v-model.number="draft[service.sku]" class="price-input" type="number" min="0" step="1" :aria-label="t('{name} 的价格', { name: service.name })" />
                  <button v-if="draft[service.sku] !== defaults[service.sku]" class="link-button" @click="draft[service.sku] = defaults[service.sku]">{{ t('恢复默认') }}</button>
                </template>
                <b v-else>{{ money(service.price) }}</b>
                <small v-if="service.price">/ {{ service.unit }}</small>
                <small v-if="!editing && defaults[service.sku] !== undefined && service.price !== defaults[service.sku]">{{ t('默认 {p}', { p: money(defaults[service.sku]) }) }}</small>
              </div>
            </header>
            <p class="service-summary">{{ service.summary }}</p>
            <ul class="check-list">
              <li v-for="item in service.includes" :key="item">{{ item }}</li>
            </ul>
            <ul v-if="service.excludes?.length" class="cross-list">
              <li v-for="item in service.excludes" :key="item">{{ item }}</li>
            </ul>
            <footer>
              <span v-if="service.revision_rounds">{{ t('{n} 轮修改', { n: service.revision_rounds }) }}</span>
              <span v-if="service.turnaround">{{ service.turnaround }}</span>
              <span v-if="service.discount_eligible" class="chip chip-spark">{{ t('参与早鸟 / 团报') }}</span>
            </footer>
          </article>
        </div>
      </section>

      <div class="policy-grid">
        <section class="workspace-card">
          <div class="card-heading"><div><h2>{{ t('优惠规则') }}</h2></div></div>
          <ul class="policy-list">
            <li>{{ catalog.discounts.early_bird.rule }}</li>
            <li>{{ catalog.discounts.group.rule }}</li>
            <li>{{ catalog.discounts.referral.rule }}</li>
            <li>{{ catalog.discounts.stacking_rule }}</li>
            <li>{{ catalog.discounts.floor_rule }}</li>
            <li><strong>{{ catalog.discounts.no_negotiation }}</strong></li>
          </ul>
        </section>
        <section class="workspace-card">
          <div class="card-heading"><div><h2>{{ t('付款与退款') }}</h2></div></div>
          <ul class="policy-list">
            <li>{{ catalog.payment.rule }}</li>
            <li v-for="rule in catalog.refund_policy.rules" :key="rule.stage"><strong>{{ rule.label }}: </strong>{{ rule.rule }}</li>
            <li v-for="note in catalog.refund_policy.notes" :key="note">{{ note }}</li>
            <li>{{ t('发票：') }}{{ catalog.invoice.type }}，{{ catalog.invoice.timing }}</li>
          </ul>
        </section>
      </div>

      <section class="catalog-group">
        <h2 class="group-title">{{ t('五国资料') }} <small>{{ countries.last_verified }} · {{ t('面向') }} {{ countries.target_intake }}</small></h2>
        <div class="country-list">
          <details v-for="country in countries.countries" :key="country.key" class="country-item">
            <summary>
              <strong>{{ country.key }}</strong>
              <span>{{ country.name_en }}</span>
              <small>{{ country.tuition_non_eu }}</small>
            </summary>
            <dl class="detail-list country-detail">
              <div><dt>{{ t('申请平台') }}</dt><dd>{{ country.portal }}</dd></div>
              <div><dt>{{ t('申请时间') }}</dt><dd>{{ country.application_window }}</dd></div>
              <div><dt>{{ t('申请费') }}</dt><dd>{{ country.application_fee }}</dd></div>
              <div><dt>{{ t('学制') }}</dt><dd>{{ country.duration }}</dd></div>
              <div><dt>{{ t('语言') }}</dt><dd>{{ country.language }}</dd></div>
              <div><dt>{{ t('居留') }}</dt><dd>{{ country.residence_permit }}</dd></div>
              <div><dt>{{ t('代表项目') }}</dt><dd>{{ country.representative_programs.join('；') }}</dd></div>
              <div><dt>{{ t('官网') }}</dt><dd><a v-for="url in country.source_urls" :key="url" :href="url" target="_blank" rel="noreferrer" class="source-link">{{ url.replace(/^https?:\/\//, '') }}</a></dd></div>
            </dl>
          </details>
        </div>
      </section>
    </template>
  </section>
</template>

<script setup>
import { t } from '../lib/i18n.js'
import { computed, onMounted, ref } from 'vue'
import { requestCatalog, updatePrices } from '../lib/backends'
import { CATEGORY_LABELS, CATEGORY_ORDER, money } from '../lib/labels'

const props = defineProps({
  settings: { type: Object, required: true },
  readonly: { type: Boolean, default: false }
})
const emit = defineEmits(['toast'])

const catalog = ref(null)
const countries = ref({ countries: [] })
const defaults = ref({})
const error = ref('')
const editing = ref(false)
const saving = ref(false)
const draft = ref({})

const groups = computed(() => {
  if (!catalog.value) return []
  return CATEGORY_ORDER
    .map((category) => ({
      category,
      label: t(CATEGORY_LABELS[category]),
      services: catalog.value.services.filter((service) => service.category === category)
    }))
    .filter((group) => group.services.length)
})

function apply(data) {
  catalog.value = data.catalog
  countries.value = data.countries
  defaults.value = data.default_prices || {}
}

function startEditing() {
  draft.value = Object.fromEntries(catalog.value.services.map((s) => [s.sku, s.price]))
  editing.value = true
}

async function savePrices() {
  const changes = Object.fromEntries(
    catalog.value.services
      .filter((s) => draft.value[s.sku] !== s.price)
      .map((s) => [s.sku, draft.value[s.sku]])
  )
  if (Object.values(changes).some((p) => !Number.isInteger(p) || p < 0)) {
    emit('toast', t('价格必须是不小于 0 的整数'))
    return
  }
  if (!Object.keys(changes).length) {
    editing.value = false
    return
  }
  saving.value = true
  try {
    const data = await updatePrices(props.settings, changes)
    apply(data)
    editing.value = false
    emit('toast', data.knowledge_synced === false ? t('价格已生效，知识库稍后同步') : t('价格已更新'))
  } catch (err) {
    emit('toast', t('保存失败：{m}', { m: err.message }))
  } finally {
    saving.value = false
  }
}

onMounted(async () => {
  try {
    apply(await requestCatalog(props.settings))
  } catch (err) {
    error.value = err.message
  }
})
</script>
