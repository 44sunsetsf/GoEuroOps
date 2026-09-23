<template>
  <section class="page page-scroll page-catalog">
    <div class="page-heading">
      <div class="heading-copy">
        <h1>服务价目</h1>
        <p>
          助手报价、退款计算和知识库都读取同一份业务目录
          <code>backend/business/catalog.yaml</code>。改价格只改这个文件，重启后端即可生效。
        </p>
      </div>
      <div v-if="catalog" class="count-display"><strong>{{ catalog.services.length }}</strong><span>项服务 · {{ catalog.catalog_version }}</span></div>
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
                <b>{{ money(service.price) }}</b>
                <small v-if="service.price">/ {{ service.unit }}</small>
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
              <span v-if="service.revision_rounds">{{ service.revision_rounds }} 轮修改</span>
              <span v-if="service.turnaround">{{ service.turnaround }}</span>
              <span v-if="service.discount_eligible" class="chip chip-spark">参与早鸟 / 团报</span>
            </footer>
          </article>
        </div>
      </section>

      <div class="policy-grid">
        <section class="workspace-card">
          <div class="card-heading"><div><h2>优惠规则</h2></div></div>
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
          <div class="card-heading"><div><h2>付款与退款</h2></div></div>
          <ul class="policy-list">
            <li>{{ catalog.payment.rule }}</li>
            <li v-for="rule in catalog.refund_policy.rules" :key="rule.stage"><strong>{{ rule.label }}：</strong>{{ rule.rule }}</li>
            <li v-for="note in catalog.refund_policy.notes" :key="note">{{ note }}</li>
            <li>发票：{{ catalog.invoice.type }}，{{ catalog.invoice.timing }}</li>
          </ul>
        </section>
      </div>

      <section class="catalog-group">
        <h2 class="group-title">五国资料 <small>{{ countries.last_verified }} · 面向 {{ countries.target_intake }}</small></h2>
        <div class="country-list">
          <details v-for="country in countries.countries" :key="country.key" class="country-item">
            <summary>
              <strong>{{ country.key }}</strong>
              <span>{{ country.name_en }}</span>
              <small>{{ country.tuition_non_eu }}</small>
            </summary>
            <dl class="detail-list country-detail">
              <div><dt>申请平台</dt><dd>{{ country.portal }}</dd></div>
              <div><dt>申请时间</dt><dd>{{ country.application_window }}</dd></div>
              <div><dt>申请费</dt><dd>{{ country.application_fee }}</dd></div>
              <div><dt>学制</dt><dd>{{ country.duration }}</dd></div>
              <div><dt>语言</dt><dd>{{ country.language }}</dd></div>
              <div><dt>居留</dt><dd>{{ country.residence_permit }}</dd></div>
              <div><dt>代表项目</dt><dd>{{ country.representative_programs.join('；') }}</dd></div>
              <div><dt>官网</dt><dd><a v-for="url in country.source_urls" :key="url" :href="url" target="_blank" rel="noreferrer" class="source-link">{{ url.replace(/^https?:\/\//, '') }}</a></dd></div>
            </dl>
          </details>
        </div>
      </section>
    </template>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { requestCatalog } from '../lib/backends'
import { CATEGORY_LABELS, CATEGORY_ORDER, money } from '../lib/labels'

const props = defineProps({ settings: { type: Object, required: true } })

const catalog = ref(null)
const countries = ref({ countries: [] })
const error = ref('')

const groups = computed(() => {
  if (!catalog.value) return []
  return CATEGORY_ORDER
    .map((category) => ({
      category,
      label: CATEGORY_LABELS[category],
      services: catalog.value.services.filter((service) => service.category === category)
    }))
    .filter((group) => group.services.length)
})

onMounted(async () => {
  try {
    const data = await requestCatalog(props.settings)
    catalog.value = data.catalog
    countries.value = data.countries
  } catch (err) {
    error.value = err.message
  }
})
</script>
